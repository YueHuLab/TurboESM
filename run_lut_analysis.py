"""
分析 K_rot 数据分布，找到最优量化策略。
"""
import torch
import torch.nn.functional as F
from transformers import EsmModel, EsmTokenizer
from esm_turbo.turbo_esm import TurboESM

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
MODEL_DIR = 'weights/esm2_650M'
CKPT = 'weights/esm2_650M_turbo.pt'
seq = 'MKTVRQERLKSIVVLGAGGVGSAVADYLRQKGIPVT'

tokenizer = EsmTokenizer.from_pretrained(MODEL_DIR)
inputs = tokenizer(seq, return_tensors='pt').to(device)
orig = EsmModel.from_pretrained(MODEL_DIR).to(device).eval()

turbo = TurboESM(model_dir=MODEL_DIR, checkpoint_path=CKPT, device=device)
turbo.kv_cache.reset()
_ = turbo.generate(seq)

# 从 layer 0 拿 K
captured = {}
def hook(module, inp, out):
    hs = inp[0]
    bsz, sl, _ = hs.shape
    nh = turbo.config.num_attention_heads
    hd = turbo.config.hidden_size // nh
    with torch.no_grad():
        q = module.query(hs).view(bsz, sl, nh, hd).transpose(1, 2)
        k = module.key(hs).view(bsz, sl, nh, hd).transpose(1, 2)
        q, k = module.rotary_embeddings(q, k)
    captured['k'] = k.detach().cpu()

h = orig.encoder.layer[0].attention.self.register_forward_hook(hook)
with torch.no_grad():
    orig(**inputs)
h.remove()

k = captured['k']  # (1, heads, seq, head_dim)
pi0 = turbo.model.encoder.layer[0].attention.self.pi_matrix.cpu()
lut0 = turbo.model.encoder.layer[0].attention.self.lut.cpu()

pi_t = pi0.transpose(-1, -2)
k_rot = torch.einsum('bhsd,hde->bhse', k, pi_t)

flat = k_rot.flatten()
print("K_rot 分布统计:")
for p in [0, 1, 5, 25, 50, 75, 95, 99, 100]:
    print(f"  p{p:3d}: {torch.quantile(flat.float(), p/100).item():.4f}")

print(f"\n标准差: {flat.std().item():.4f}")
print(f"均值:   {flat.mean().item():.4f}")

# 离群值占比
total = flat.numel()
out3 = ((flat.abs() > 3 * flat.std()).sum() / total * 100).item()
out5 = ((flat.abs() > 5 * flat.std()).sum() / total * 100).item()
print(f"\n|x| > 3σ 占比: {out3:.2f}%")
print(f"|x| > 5σ 占比: {out5:.2f}%")

print(f"\n当前 LUT (head 0): {lut0[0].tolist()}")

# 评估不同截断策略下的量化 RMSE
print("\n不同截断策略的量化误差:")
import scipy.cluster.vq as vq
import numpy as np

for clip_pct in [100, 99.9, 99, 98, 95]:
    lo = torch.quantile(flat.float(), (100-clip_pct)/200).item()
    hi = torch.quantile(flat.float(), 1-(100-clip_pct)/200).item()
    clipped = flat.float().numpy()
    clipped_in = clipped[(clipped >= lo) & (clipped <= hi)]
    cents, _ = vq.kmeans(clipped_in, 8)
    cents = np.sort(cents)
    # 对全量数据做量化（不截断）
    cents_t = torch.tensor(cents, dtype=torch.float32)
    dist = (flat.float().unsqueeze(-1) - cents_t).abs()
    idx = dist.argmin(dim=-1)
    recon = cents_t[idx]
    rmse = (flat.float() - recon).pow(2).mean().sqrt().item()
    snr = flat.float().pow(2).mean().sqrt().item() / rmse
    print(f"  clip p{(100-clip_pct)/2:.1f}~p{100-(100-clip_pct)/2:.1f}: RMSE={rmse:.4f}, SNR={snr:.1f}x, LUT=[{cents[0]:.3f}, ..., {cents[-1]:.3f}]")
