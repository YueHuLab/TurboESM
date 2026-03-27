"""
对比 K 和 V 的分布差异，验证用同一 LUT 量化 V 的误差。
"""
import torch
import torch.nn.functional as F
from transformers import EsmModel, EsmTokenizer
from esm_turbo.turbo_esm import TurboESM
import scipy.cluster.vq as vq
import numpy as np

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
MODEL_DIR = 'weights/esm2_650M'
CKPT = 'weights/esm2_650M_turbo.pt'
seq = 'MKTVRQERLKSIVVLGAGGVGSAVADYLRQKGIPVT'

tokenizer = EsmTokenizer.from_pretrained(MODEL_DIR)
inputs = tokenizer(seq, return_tensors='pt').to(device)
orig = EsmModel.from_pretrained(MODEL_DIR).to(device).eval()
turbo = TurboESM(model_dir=MODEL_DIR, checkpoint_path=CKPT, device=device)

captured = {}
def hook(module, inp, out):
    hs = inp[0]
    bsz, sl, _ = hs.shape
    nh = turbo.config.num_attention_heads
    hd = turbo.config.hidden_size // nh
    with torch.no_grad():
        q = module.query(hs).view(bsz, sl, nh, hd).transpose(1, 2)
        k = module.key(hs).view(bsz, sl, nh, hd).transpose(1, 2)
        v = module.value(hs).view(bsz, sl, nh, hd).transpose(1, 2)
        q_scaled = q * (hd ** -0.5)
        q_rope, k_rope = module.rotary_embeddings(q_scaled, k)
    captured['q'] = q_rope.detach().cpu()
    captured['k'] = k_rope.detach().cpu()
    captured['v'] = v.detach().cpu()

h = orig.encoder.layer[0].attention.self.register_forward_hook(hook)
with torch.no_grad():
    orig(**inputs)
h.remove()

q_orig = captured['q']
k_orig = captured['k']
v_orig = captured['v']

pi0 = turbo.model.encoder.layer[0].attention.self.pi_matrix.cpu()
lut0 = turbo.model.encoder.layer[0].attention.self.lut.cpu()
pi_t = pi0.transpose(-1, -2)
k_rot = torch.einsum('bhsd,hde->bhse', k_orig, pi_t)
q_rot = torch.einsum('bhsd,hde->bhse', q_orig, pi_t)

k_flat = k_rot.flatten().float()
v_flat = v_orig.flatten().float()

print("K_rot 分布: min={:.3f}, max={:.3f}, std={:.3f}".format(
    k_flat.min().item(), k_flat.max().item(), k_flat.std().item()))
print("V     分布: min={:.3f}, max={:.3f}, std={:.3f}".format(
    v_flat.min().item(), v_flat.max().item(), v_flat.std().item()))

# 用 K 的 LUT 量化 V 的误差
cents_k = lut0[0]  # head 0 的 LUT
dist_v = (v_flat.unsqueeze(-1) - cents_k).abs()
idx_v = dist_v.argmin(dim=-1)
v_quant_with_k_lut = cents_k[idx_v]
rmse_v_with_k = (v_flat - v_quant_with_k_lut).pow(2).mean().sqrt().item()
snr_v_with_k = v_flat.pow(2).mean().sqrt().item() / rmse_v_with_k
print(f"\n用 K 的 LUT 量化 V: RMSE={rmse_v_with_k:.4f}, SNR={snr_v_with_k:.1f}x")

# 用 V 自己的最优 LUT 量化 V
cents_v, _ = vq.kmeans(v_flat.numpy(), 8)
cents_v = torch.tensor(np.sort(cents_v), dtype=torch.float32)
dist_v2 = (v_flat.unsqueeze(-1) - cents_v).abs()
idx_v2 = dist_v2.argmin(dim=-1)
v_quant_optimal = cents_v[idx_v2]
rmse_v_opt = (v_flat - v_quant_optimal).pow(2).mean().sqrt().item()
snr_v_opt = v_flat.pow(2).mean().sqrt().item() / rmse_v_opt
print(f"用 V 的最优 LUT 量化 V: RMSE={rmse_v_opt:.4f}, SNR={snr_v_opt:.1f}x")

# 对比 attention 相似度
ref_attn = F.scaled_dot_product_attention(q_rot, k_rot, v_orig, scale=1.0)

# 当前做法：K 和 V 都用同一 LUT
k_quant = lut0[torch.argmin((k_rot.float().unsqueeze(-1) - lut0).abs(), dim=-1)]
v_quant_bad = lut0[torch.argmin((v_orig.float().unsqueeze(-1) - lut0).abs(), dim=-1)]
attn_bad = F.scaled_dot_product_attention(q_rot, k_quant, v_quant_bad, scale=1.0)
sim_bad = F.cosine_similarity(ref_attn.float(), attn_bad.float(), dim=-1).mean().item()
print(f"\n当前做法（K/V 同一 LUT）: Attn 相似度={sim_bad:.4f}")

# 改进：K 用 K-LUT，V 用独立 V-LUT
v_quant_good = cents_v[torch.argmin((v_orig.float().unsqueeze(-1) - cents_v).abs(), dim=-1)]
attn_good = F.scaled_dot_product_attention(q_rot, k_quant, v_quant_good, scale=1.0)
sim_good = F.cosine_similarity(ref_attn.float(), attn_good.float(), dim=-1).mean().item()
print(f"改进做法（K/V 独立 LUT）: Attn 相似度={sim_good:.4f}")

print(f"\nV 的最优 LUT: {cents_v.tolist()}")
print(f"K 的当前 LUT: {lut0[0].tolist()}")
