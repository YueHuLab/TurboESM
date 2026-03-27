"""
性能基准测试：对比原始模型 vs TurboESM 的推理速度。
分别测试 Prefill 和模拟 Decode 场景。
"""
import torch
import time
from transformers import EsmModel, EsmTokenizer
from esm_turbo.turbo_esm import TurboESM

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
MODEL_DIR = 'weights/esm2_650M'
CKPT = 'weights/esm2_650M_turbo.pt'
WARMUP = 3
RUNS = 10

seq_short = "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"          # ~30 tokens
seq_long  = "VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"  # ~140 tokens

tokenizer = EsmTokenizer.from_pretrained(MODEL_DIR)
orig  = EsmModel.from_pretrained(MODEL_DIR).to(device).eval()
turbo = TurboESM(model_dir=MODEL_DIR, checkpoint_path=CKPT, device=device)

def bench(fn, warmup=WARMUP, runs=RUNS):
    for _ in range(warmup):
        fn()
    if device == 'mps':
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    if device == 'mps':
        torch.mps.synchronize()
    return (time.perf_counter() - t0) / runs * 1000  # ms

print(f"设备: {device}  (warmup={WARMUP}, runs={RUNS})\n")
print(f"{'场景':<30} {'原始模型 ms':>12} {'TurboESM ms':>13} {'加速比':>8}")
print("-" * 68)

for label, seq in [("Prefill 短序列(~30tok)", seq_short),
                   ("Prefill 长序列(~140tok)", seq_long)]:
    inputs = tokenizer(seq, return_tensors='pt').to(device)

    with torch.no_grad():
        t_orig  = bench(lambda: orig(**inputs))
        turbo.kv_cache.reset()
        t_turbo = bench(lambda: turbo.generate(seq))

    print(f"{label:<30} {t_orig:>11.1f}  {t_turbo:>12.1f}  {t_orig/t_turbo:>7.2f}x")

# 分析各组件耗时
print("\n── Prefill 内部各阶段耗时分解（长序列）──")
inputs_long = tokenizer(seq_long, return_tensors='pt').to(device)

# 1. 纯 forward（无 KV cache 存储）
with torch.no_grad():
    t_fwd = bench(lambda: orig(**inputs_long))

# 2. quantize + pack 耗时
from esm_turbo.kv_cache import TurboKVCache
from esm_turbo.modeling_esm_turbo import TurboEsmSelfAttention
attn0 = turbo.model.encoder.layer[0].attention.self
config = turbo.config
kv = TurboKVCache(config, 1, 512, device)

# 抓一个真实 k_rot
captured = {}
def hook(m, inp, out):
    hs = inp[0]
    bsz, sl, _ = hs.shape
    nh = config.num_attention_heads
    hd = config.hidden_size // nh
    with torch.no_grad():
        k = m.key(hs).view(bsz, sl, nh, hd).transpose(1, 2)
        q = m.query(hs).view(bsz, sl, nh, hd).transpose(1, 2)
        q, k = m.rotary_embeddings(q, k)
        pi_t = attn0.pi_matrix.transpose(-1, -2)
        k_rot = torch.einsum('bhsd,hde->bhse', k, pi_t)
        v = m.value(hs).view(bsz, sl, nh, hd).transpose(1, 2)
    captured['k_rot'] = k_rot
    captured['v'] = v

h = orig.encoder.layer[0].attention.self.register_forward_hook(hook)
with torch.no_grad():
    orig(**inputs_long)
h.remove()

k_rot = captured['k_rot']
v     = captured['v']

def do_store():
    kv.reset()
    kv.store_packed(0, k_rot, v, attn0.lut_k, attn0.lut_v, torch.sign(k_rot))

def do_fetch():
    kv.fetch_unpacked(0, attn0.lut_k.cpu(), attn0.lut_v.cpu())

t_store = bench(do_store)
kv.reset()
kv.store_packed(0, k_rot, v, attn0.lut_k, attn0.lut_v, torch.sign(k_rot))
t_fetch = bench(do_fetch)

print(f"  原始模型 forward:          {t_fwd:.1f} ms")
print(f"  store_packed (单层):       {t_store:.2f} ms  ← Python pack 循环")
print(f"  fetch_unpacked (单层):     {t_fetch:.2f} ms  ← Python unpack 循环")
print(f"  store × {config.num_hidden_layers} 层估算:        {t_store * config.num_hidden_layers:.1f} ms")
print(f"  fetch × {config.num_hidden_layers} 层估算:        {t_fetch * config.num_hidden_layers:.1f} ms")
print()
print("慢的根本原因:")
print("  1. _pack_3bit/_unpack_3bit 用 Python for 循环（8次），应用 torch bitwise 向量化")
print("  2. _quantize 用 unsqueeze+broadcast 做 nearest-neighbor，O(seq*head_dim*8)，应用 torch.bucketize")
print("  3. fetch_unpacked 在 CPU 上跑（lut 传到 cpu），MPS 张量传回 CPU 有额外拷贝开销")
print("  4. Prefill 阶段存 cache 但不用 cache，存储是纯额外开销")
