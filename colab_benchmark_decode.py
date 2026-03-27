"""
Colab CUDA Decode Benchmark：对比 FP32 KV Cache vs 3-bit TurboKV Cache 的 Decode 延迟与显存。

模拟场景：先 Prefill 一段上下文，再逐 token Decode（seq_len=1）。
这才是 TurboQuant 真正的目标场景。
"""
import glob, shutil, sys, torch, time
import torch.nn.functional as F
from transformers import EsmModel, EsmTokenizer

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
WARMUP      = 5
RUNS        = 50

from esm_turbo.turbo_esm import TurboESM
from esm_turbo.kv_cache import TurboKVCache
from esm_turbo.triton_kernels import TRITON_AVAILABLE

print(f"Triton 可用: {TRITON_AVAILABLE}")

tokenizer = EsmTokenizer.from_pretrained(MODEL_LOCAL)
ref_model = EsmModel.from_pretrained(MODEL_LOCAL).to(DEVICE).eval()
turbo     = TurboESM(model_dir=MODEL_LOCAL, checkpoint_path=CKPT_PATH, device=DEVICE)
config    = turbo.config

# ── 工具 ─────────────────────────────────────────────────────────────────────
def bench_ms(fn, warmup=WARMUP, runs=RUNS):
    with torch.no_grad():
        for _ in range(warmup): fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs): fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000

def peak_mb(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.no_grad(): fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 1024 / 1024

# ── 场景 1：fetch_unpacked 延迟（Triton vs PyTorch）─────────────────────────
print("\n" + "=" * 65)
print("  场景 1：fetch_unpacked 单次延迟（Triton kernel vs PyTorch）")
print("=" * 65)

# 先做一次 Prefill 填满 cache
prefill_seq = "VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
turbo.kv_cache.reset()
with torch.no_grad():
    turbo.generate(prefill_seq)

attn0  = turbo.model.encoder.layer[0].attention.self
lut_k  = attn0.lut_k
lut_v  = attn0.lut_v
res_sc = attn0.residual_scale_k
valid_len = turbo.kv_cache.layer_seq_len[0, 0].item()
print(f"  Cache 已填充长度: {valid_len} tokens")

# PyTorch 路径（强制关闭 Triton）
import esm_turbo.triton_kernels as tk
_orig_available = tk.TRITON_AVAILABLE

def fetch_pytorch():
    tk.TRITON_AVAILABLE = False
    turbo.kv_cache.fetch_unpacked(0, lut_k, lut_v, res_sc)
    tk.TRITON_AVAILABLE = _orig_available

def fetch_triton():
    turbo.kv_cache.fetch_unpacked(0, lut_k, lut_v, res_sc)

t_pytorch = bench_ms(fetch_pytorch)
t_triton  = bench_ms(fetch_triton) if TRITON_AVAILABLE else float('nan')

print(f"  PyTorch fetch_unpacked: {t_pytorch:.3f} ms")
if TRITON_AVAILABLE:
    print(f"  Triton  fetch_unpacked: {t_triton:.3f} ms  ({t_pytorch/t_triton:.2f}x 加速)")
else:
    print("  Triton 不可用")

# ── 场景 2：模拟 Decode 步骤端到端延迟 ───────────────────────────────────────
print("\n" + "=" * 65)
print("  场景 2：单步 Decode 端到端延迟")
print("  （Prefill 上下文 → 追加 1 个 token → 得到输出）")
print("=" * 65)

context_seqs = {
    "短上下文  (~32 tok)":  "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
    "中上下文  (~143 tok)": prefill_seq,
}

# 构造单 token 输入（用 [MASK] 模拟 decode token）
single_token = tokenizer("M", return_tensors="pt", add_special_tokens=False).to(DEVICE)

print(f"\n{'场景':22} {'上下文':>8}  {'FP32 Decode(ms)':>16} {'Turbo Decode(ms)':>17} {'加速':>6}  {'FP32 MB':>8} {'Turbo MB':>9}")
print("-" * 90)

for label, ctx_seq in context_seqs.items():
    ctx_inputs = tokenizer(ctx_seq, return_tensors="pt").to(DEVICE)
    ctx_len    = ctx_inputs['input_ids'].shape[1]

    # ── FP32 baseline：完整上下文 + 1 token 重新推理（无 cache）
    full_ids = torch.cat([ctx_inputs['input_ids'],
                          single_token['input_ids']], dim=1)
    full_mask = torch.ones_like(full_ids)
    full_inputs = {'input_ids': full_ids, 'attention_mask': full_mask}

    def ref_decode():
        ref_model(**full_inputs)

    t_ref = bench_ms(ref_decode)
    m_ref = peak_mb(ref_decode)

    # ── Turbo：Prefill 一次，然后 Decode
    def turbo_prefill_decode():
        turbo.kv_cache.reset()
        turbo.generate(ctx_seq)          # Prefill
        turbo.generate("M")              # Decode (seq_len=1 触发 cache)

    # 先做一次 prefill 填 cache，然后只测 decode 部分
    turbo.kv_cache.reset()
    with torch.no_grad():
        turbo.generate(ctx_seq)

    def turbo_decode_only():
        turbo.generate("M")

    t_turbo = bench_ms(turbo_decode_only)
    m_turbo = peak_mb(turbo_decode_only)

    speedup = t_ref / t_turbo

    print(f"{label:22} {ctx_len:>8}  {t_ref:>16.3f} {t_turbo:>17.3f} {speedup:>5.2f}x  {m_ref:>8.2f} {m_turbo:>9.2f}")

# ── 场景 3：KV Cache 静态显存对比 ────────────────────────────────────────────
print("\n" + "=" * 65)
print("  场景 3：KV Cache 静态显存（max_seq_len=1024）")
print("=" * 65)

num_layers = config.num_hidden_layers
num_heads  = config.num_attention_heads
head_dim   = config.hidden_size // num_heads
max_seq    = 1024

fp32_mb  = 2 * num_layers * 1 * num_heads * max_seq * head_dim * 4 / 1024 / 1024
bit3_mb  = 2 * num_layers * 1 * num_heads * max_seq * (head_dim // 8) * 4 / 1024 / 1024
qjl_mb   = num_layers * 1 * num_heads * max_seq * (head_dim // 32) * 4 / 1024 / 1024
turbo_total = bit3_mb + qjl_mb

print(f"  配置: {num_layers}层 × {num_heads}头 × head_dim={head_dim}, max_seq={max_seq}")
print(f"  FP32  KV Cache: {fp32_mb:.1f} MB")
print(f"  3-bit KV Cache: {bit3_mb:.1f} MB")
print(f"  1-bit QJL:      {qjl_mb:.1f} MB")
print(f"  Turbo 合计:     {turbo_total:.1f} MB  （压缩比 {fp32_mb/turbo_total:.1f}x）")

# 实际分配显存
torch.cuda.reset_peak_memory_stats()
base = torch.cuda.memory_allocated()
dummy_fp32 = torch.zeros(2, num_layers, 1, num_heads, max_seq, head_dim,
                          dtype=torch.float32, device=DEVICE)
actual_fp32 = (torch.cuda.memory_allocated() - base) / 1024 / 1024
del dummy_fp32

base = torch.cuda.memory_allocated()
dummy_kv = TurboKVCache(config, max_batch_size=1, max_seq_len=max_seq, device=DEVICE)
actual_turbo = (torch.cuda.memory_allocated() - base) / 1024 / 1024
del dummy_kv

print(f"\n  实际分配（CUDA）:")
print(f"  FP32  KV Cache: {actual_fp32:.1f} MB")
print(f"  TurboKVCache:   {actual_turbo:.1f} MB  （实际压缩比 {actual_fp32/actual_turbo:.1f}x）")
