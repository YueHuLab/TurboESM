"""
验证 turbo_fused_decode_attention kernel 正确性。
对比两条路径的输出：
  路径 A（当前生产路径）: fetch_unpacked → scaled_dot_product_attention
  路径 B（fused kernel）: turbo_fused_decode_attention（一步完成）
两者输出应高度一致（cosine sim > 0.999）。
"""
import glob, shutil, sys, torch
import torch.nn.functional as F

# ── 环境初始化 ───────────────────────────────────────────────────────────────
shutil.rmtree('/content/esm_turbo/__pycache__', ignore_errors=True)
for k in [k for k in sys.modules if 'esm_turbo' in k]:
    del sys.modules[k]

candidates = glob.glob(
    '/root/.cache/huggingface/hub/models--facebook--esm2_t33_650M_UR50D/snapshots/*/config.json'
)
MODEL_LOCAL = candidates[0].replace('/config.json', '')
CKPT_PATH   = '/content/weights/esm2_650M_turbo.pt'
DEVICE      = 'cuda'

from esm_turbo.turbo_esm import TurboESM
from esm_turbo.triton_kernels import TRITON_AVAILABLE, turbo_fused_decode_attention
import esm_turbo.triton_kernels as tk

print(f"Triton 可用: {TRITON_AVAILABLE}")
if not TRITON_AVAILABLE:
    raise RuntimeError("需要 CUDA + Triton 环境")

from transformers import EsmTokenizer
tokenizer = EsmTokenizer.from_pretrained(MODEL_LOCAL)
turbo     = TurboESM(model_dir=MODEL_LOCAL, checkpoint_path=CKPT_PATH, device=DEVICE)
config    = turbo.config

# ── 填充 KV Cache（Prefill 阶段）────────────────────────────────────────────
test_cases = {
    "短序列  (32 tok)":  "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
    "中序列  (143 tok)": "VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
    "长序列  (165 tok)": "MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQSSSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNS",
}

print("\n" + "=" * 70)
print("  fused kernel vs 两步走 输出一致性验证（逐层 × 逐序列）")
print("=" * 70)

for seq_label, ctx_seq in test_cases.items():
    # Prefill 填 cache
    turbo.kv_cache.reset()
    with torch.no_grad():
        turbo.generate(ctx_seq)

    ctx_len = turbo.kv_cache.layer_seq_len[0, 0].item()

    # 构造 Decode 阶段的 Q（用第一层 attention 的真实 Q）
    # 取 layer 0 的参数作为代表
    attn0    = turbo.model.encoder.layer[0].attention.self
    lut_k0   = attn0.lut_k
    lut_v0   = attn0.lut_v
    res_sc0  = attn0.residual_scale_k
    head_dim = config.hidden_size // config.num_attention_heads
    num_heads = config.num_attention_heads

    # 随机生成一个合理量级的 Q（模拟 decode token 经过 RoPE+Pi 后的 Q）
    torch.manual_seed(42)
    q_decode = torch.randn(1, num_heads, 1, head_dim, device=DEVICE, dtype=torch.float32)
    q_decode = q_decode * (head_dim ** -0.5)  # 与 forward 里的预缩放一致

    # ── 路径 A：两步走（当前生产路径）──────────────────────────────────────
    with torch.no_grad():
        # 强制走 PyTorch 路径做 fetch
        k_fp, v_fp = turbo.kv_cache.fetch_unpacked(0, lut_k0, lut_v0, res_sc0)
        out_A = F.scaled_dot_product_attention(
            q_decode, k_fp, v_fp, attn_mask=None, scale=1.0
        )  # (1, num_heads, 1, head_dim)

    # ── 路径 B：fused decode attention kernel ───────────────────────────────
    valid_len = turbo.kv_cache.layer_seq_len[0, 0].item()
    k_packed  = turbo.kv_cache.k_cache[0, :, :, :valid_len, :]
    v_packed  = turbo.kv_cache.v_cache[0, :, :, :valid_len, :]
    qjl       = turbo.kv_cache.qjl_cache[0, :, :, :valid_len, :]

    with torch.no_grad():
        out_B = turbo_fused_decode_attention(
            q_decode, k_packed, v_packed, qjl,
            lut_k0, lut_v0, res_sc0
        )  # (1, num_heads, 1, head_dim)

    # ── 数值对比 ────────────────────────────────────────────────────────────
    # 展平为 (num_heads, head_dim) 做逐 head cosine sim
    a = out_A.squeeze(0).squeeze(1)   # (num_heads, head_dim)
    b = out_B.squeeze(0).squeeze(1)

    cos_per_head = F.cosine_similarity(a, b, dim=-1)  # (num_heads,)
    cos_mean  = cos_per_head.mean().item()
    cos_min   = cos_per_head.min().item()
    abs_diff  = (a - b).abs().max().item()

    status = "✅" if cos_mean > 0.999 else ("⚠️" if cos_mean > 0.99 else "❌")
    print(f"\n{status} {seq_label}  (cache={ctx_len} tok)")
    print(f"   cosine sim  均值: {cos_mean:.6f}   最差 head: {cos_min:.6f}")
    print(f"   最大绝对误差:     {abs_diff:.2e}")

    # 打印最差的 3 个 head
    worst_heads = cos_per_head.argsort()[:3]
    print(f"   最差 3 个 head: {[(int(h), f'{cos_per_head[h]:.4f}') for h in worst_heads]}")

# ── 多层验证（用中序列，逐层检查）────────────────────────────────────────────
print("\n" + "=" * 70)
print("  逐层验证（中序列，33层）")
print("=" * 70)

ctx_seq = test_cases["中序列  (143 tok)"]
turbo.kv_cache.reset()
with torch.no_grad():
    turbo.generate(ctx_seq)

torch.manual_seed(42)
head_dim  = config.hidden_size // config.num_attention_heads
num_heads = config.num_attention_heads
q_decode  = torch.randn(1, num_heads, 1, head_dim, device=DEVICE) * (head_dim ** -0.5)

print(f"{'Layer':>6}  {'cos_mean':>10}  {'cos_min':>10}  {'max_abs_err':>12}  {'状态':>4}")
print("-" * 52)

all_pass = True
for layer_idx in range(config.num_hidden_layers):
    attn   = turbo.model.encoder.layer[layer_idx].attention.self
    lut_k  = attn.lut_k
    lut_v  = attn.lut_v
    res_sc = attn.residual_scale_k

    valid_len = turbo.kv_cache.layer_seq_len[layer_idx, 0].item()
    k_packed  = turbo.kv_cache.k_cache[layer_idx, :, :, :valid_len, :]
    v_packed  = turbo.kv_cache.v_cache[layer_idx, :, :, :valid_len, :]
    qjl       = turbo.kv_cache.qjl_cache[layer_idx, :, :, :valid_len, :]

    with torch.no_grad():
        k_fp, v_fp = turbo.kv_cache.fetch_unpacked(layer_idx, lut_k, lut_v, res_sc)
        out_A = F.scaled_dot_product_attention(q_decode, k_fp, v_fp, scale=1.0)
        out_B = turbo_fused_decode_attention(q_decode, k_packed, v_packed, qjl, lut_k, lut_v, res_sc)

    a = out_A.squeeze(0).squeeze(1)
    b = out_B.squeeze(0).squeeze(1)
    cos   = F.cosine_similarity(a, b, dim=-1)
    ok    = cos.mean().item() > 0.999
    if not ok: all_pass = False
    mark  = "✅" if ok else "❌"
    print(f"{layer_idx:>6}  {cos.mean().item():>10.6f}  {cos.min().item():>10.6f}  {(a-b).abs().max().item():>12.2e}  {mark}")

print()
if all_pass:
    print("✅ 全部 33 层验证通过，fused kernel 数学等价，可以安全接入生产路径。")
else:
    print("❌ 存在不一致层，需要检查 fused kernel 的解包逻辑。")
