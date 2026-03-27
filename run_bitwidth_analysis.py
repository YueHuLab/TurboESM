"""
验证 4-bit vs 3-bit 量化的理论精度上限。
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
pi_t = pi0.transpose(-1, -2)
k_rot = torch.einsum('bhsd,hde->bhse', k_orig, pi_t)
q_rot = torch.einsum('bhsd,hde->bhse', q_orig, pi_t)
flat = k_rot.flatten().float().numpy()

# 基准 attention
ref_attn = F.scaled_dot_product_attention(q_rot, k_rot, v_orig, scale=1.0)

print("不同量化位宽的 Decode 相似度上限:")
print(f"{'位宽':>6} {'LUT点数':>8} {'K RMSE':>10} {'SNR':>8} {'Attn 相似度':>14}")
print("-" * 55)

for n_bits in [3, 4, 5, 8]:
    n_centroids = 2 ** n_bits
    cents, _ = vq.kmeans(flat, n_centroids)
    cents = torch.tensor(np.sort(cents), dtype=torch.float32)

    dist = (k_rot.float().unsqueeze(-1) - cents).abs()
    idx = dist.argmin(dim=-1)
    k_quant = cents[idx]

    rmse = (k_rot.float() - k_quant).pow(2).mean().sqrt().item()
    snr = k_rot.float().pow(2).mean().sqrt().item() / rmse

    turbo_attn = F.scaled_dot_product_attention(q_rot, k_quant, v_orig, scale=1.0)
    sim = F.cosine_similarity(ref_attn.float(), turbo_attn.float(), dim=-1).mean().item()

    print(f"{n_bits:>6}  {n_centroids:>8}  {rmse:>10.4f}  {snr:>7.1f}x  {sim:>13.4f}")
