"""
逐步诊断 TurboQuant 精度损失来源。
分三个阶段：
  1. 无量化（仅 Pi 旋转）-> 相似度应接近 1.0
  2. 有量化但走原生 attention（仅检查 KV 重建误差）
  3. 完整 Turbo 路径
"""
import torch
import torch.nn.functional as F
from transformers import EsmModel, EsmTokenizer
from esm_turbo.modeling_esm_turbo import inject_turbo_attention, TurboEsmSelfAttention
from esm_turbo.kv_cache import TurboKVCache
from esm_turbo.turbo_esm import TurboESM

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
seq = 'MKTVRQERLKSIVVLGAGGVGSAVADYLRQKGIPVT'
MODEL_DIR = 'weights/esm2_650M'
CKPT = 'weights/esm2_650M_turbo.pt'

tokenizer = EsmTokenizer.from_pretrained(MODEL_DIR)
inputs = tokenizer(seq, return_tensors='pt').to(device)

# ── 基准：原始模型 ──────────────────────────────────────────────
orig = EsmModel.from_pretrained(MODEL_DIR).to(device).eval()
with torch.no_grad():
    ref = orig(**inputs).last_hidden_state
print(f"原始输出 shape: {ref.shape}, dtype: {ref.dtype}")

# ── 阶段 1：检查 pi_matrix / lut 是否正确加载 ─────────────────
state_dict = torch.load(CKPT, weights_only=True, map_location=device)
for k in state_dict:
    if k.endswith('pi_matrix') or k.endswith('lut'):
        state_dict[k] = state_dict[k].float()

# 取第 0 层检查
pi0 = state_dict['encoder.layer.0.attention.self.pi_matrix']
lut0 = state_dict['encoder.layer.0.attention.self.lut']
print(f"\n[诊断] Layer0 pi_matrix shape={pi0.shape}, dtype={pi0.dtype}")
pi0_cpu = pi0.cpu()
print(f"  是单位矩阵吗? {torch.allclose(pi0_cpu, torch.eye(pi0_cpu.shape[-1]).unsqueeze(0).expand_as(pi0_cpu), atol=1e-3)}")
print(f"  pi_matrix 与其转置之积（应接近单位阵）的最大误差: {(pi0_cpu @ pi0_cpu.transpose(-1,-2) - torch.eye(pi0_cpu.shape[-1])).abs().max().item():.4f}")
print(f"[诊断] Layer0 lut shape={lut0.shape}, values=\n{lut0[0]}")

# ── 阶段 2：仅 Pi 旋转，不量化 ────────────────────────────────
print("\n[阶段2] 仅 Pi 旋转，KV 不量化...")

class NoquantTurboAttn(TurboEsmSelfAttention):
    """跳过量化，直接用旋转后的原始 KV 做 attention"""
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        hidden_states = hidden_states.to(self.query.weight.dtype)
        bsz, seq_len, _ = hidden_states.size()

        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))

        q = q * (self.attention_head_size ** -0.5)

        if hasattr(self, "rotary_embeddings"):
            emb = self.rotary_embeddings
            if emb._cos_cached is None:
                emb(q, k)
            cos = emb._cos_cached[:, :, :seq_len, :].to(q.dtype)
            sin = emb._sin_cached[:, :, :seq_len, :].to(q.dtype)
            from esm_turbo.modeling_esm_turbo import apply_rotary_pos_emb
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)

        pi_t = self.pi_matrix.transpose(-1, -2)
        q_rot = torch.einsum('bhsd,hde->bhse', q, pi_t)
        k_rot = torch.einsum('bhsd,hde->bhse', k, pi_t)

        attn_output = torch.nn.functional.scaled_dot_product_attention(q_rot, k_rot, v, attn_mask=attention_mask, scale=1.0)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return attn_output, None

from transformers.models.esm.modeling_esm import EsmSelfAttention
model_nq = EsmModel.from_pretrained(MODEL_DIR).to(device).eval()
for name, module in model_nq.named_modules():
    if isinstance(module, EsmSelfAttention) and not isinstance(module, NoquantTurboAttn):
        parent = model_nq.get_submodule(name.rsplit('.', 1)[0])
        attn = NoquantTurboAttn(model_nq.config)
        attn.load_state_dict(module.state_dict(), strict=False)
        setattr(parent, 'self', attn)

for name, module in model_nq.named_modules():
    if isinstance(module, NoquantTurboAttn):
        for k, v2 in state_dict.items():
            parts = k.split('.')
            # 找到对应层的 pi_matrix / lut
            if 'attention' in parts and 'self' in parts:
                layer_num = parts[2]
                param_name = parts[-1]
                if f'encoder.layer.{layer_num}.attention.self' in k:
                    if param_name in ('pi_matrix', 'lut'):
                        pass  # 用 load_state_dict 批量加载

model_nq.load_state_dict(state_dict, strict=False)
model_nq.to(device).eval()

with torch.no_grad():
    out_nq = model_nq(**inputs).last_hidden_state
sim_nq = F.cosine_similarity(ref.float(), out_nq.float(), dim=-1).mean().item()
print(f"  仅 Pi 旋转相似度: {sim_nq:.4f}  (应接近 1.0)")

# ── 阶段 3：完整 Turbo（含量化）──────────────────────────────
print("\n[阶段3] 完整 Turbo 路径（含 3-bit 量化）...")
turbo = TurboESM(model_dir=MODEL_DIR, checkpoint_path=CKPT, device=device)
out_turbo = turbo.generate(seq)
sim_turbo = F.cosine_similarity(ref.float(), out_turbo.float(), dim=-1).mean().item()
print(f"  完整 Turbo 相似度: {sim_turbo:.4f}  (目标 >0.99)")

# ── 阶段 4：单层 KV 量化重建误差 ──────────────────────────────
print("\n[阶段4] 检查 Layer0 KV 量化重建误差...")
config = orig.config
kv_cache = TurboKVCache(config, max_batch_size=1, max_seq_len=512, device='cpu')

# 抓一个真实的 k_rot
captured = {}
def hook_fn(module, inp, out):
    hs = inp[0].detach().cpu().float()
    bsz, sl, _ = hs.shape
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads
    k_raw = module.key(hs.to(device))
    k_t = k_raw.view(bsz, sl, num_heads, head_dim).transpose(1, 2)
    captured['k'] = k_t.cpu()

h = orig.encoder.layer[0].attention.self.register_forward_hook(hook_fn)
with torch.no_grad():
    orig(**inputs)
h.remove()

k_sample = captured['k']  # (1, heads, seq, head_dim)
pi0_cpu = pi0.cpu()
lut0_cpu = lut0.cpu()

# 旋转
pi_t = pi0_cpu.transpose(-1, -2)
k_rot = torch.einsum('bhsd,hde->bhse', k_sample, pi_t)

# 量化
x_flat = k_rot.reshape(1, config.num_attention_heads, -1)
dist = (x_flat.unsqueeze(-1) - lut0_cpu.unsqueeze(1)).abs()
indices = dist.argmin(dim=-1)
k_quant_vals = lut0_cpu[torch.arange(config.num_attention_heads).unsqueeze(0).unsqueeze(-1), indices]
k_quant_vals = k_quant_vals.view_as(k_rot)

recon_err = (k_rot - k_quant_vals).pow(2).mean().sqrt().item()
signal_rms = k_rot.pow(2).mean().sqrt().item()
print(f"  K_rot RMS: {signal_rms:.4f}")
print(f"  量化重建 RMSE: {recon_err:.4f}")
print(f"  信噪比 (signal/error): {signal_rms/recon_err:.1f}x")
print(f"  LUT 区间: [{lut0_cpu[0].min().item():.4f}, {lut0_cpu[0].max().item():.4f}]")
print(f"  K_rot 区间: [{k_rot.min().item():.4f}, {k_rot.max().item():.4f}]")

print("\n诊断完成。")
