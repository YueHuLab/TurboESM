"""
Colab CUDA Benchmark：原始 ESM-2 650M vs TurboESM
测试指标：推理速度 (ms) + 显存占用 (MB)
场景：Prefill（全序列一次性推理）
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
RUNS        = 20

from esm_turbo.turbo_esm import TurboESM

tokenizer = EsmTokenizer.from_pretrained(MODEL_LOCAL)
ref_model = EsmModel.from_pretrained(MODEL_LOCAL).to(DEVICE).eval()
turbo     = TurboESM(model_dir=MODEL_LOCAL, checkpoint_path=CKPT_PATH, device=DEVICE)

# ── 工具函数 ─────────────────────────────────────────────────────────────────
def measure_memory_mb(fn):
    """调用 fn() 前后测量显存峰值增量 (MB)"""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    with torch.no_grad():
        fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return (peak - baseline) / 1024 / 1024

def bench_ms(fn, warmup=WARMUP, runs=RUNS):
    """GPU 计时，返回平均延迟 (ms)"""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000

# ── 测试序列 ─────────────────────────────────────────────────────────────────
test_sequences = {
    "短肽  (~32 tok)":  "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
    "中等  (~143 tok)": "VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
    "长序列 (~165 tok)": "MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQSSSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNS",
}

# ── 静态显存：模型参数本身占用 ────────────────────────────────────────────────
torch.cuda.synchronize()
ref_param_mb   = sum(p.numel() * p.element_size() for p in ref_model.parameters()) / 1024 / 1024
turbo_param_mb = sum(p.numel() * p.element_size() for p in turbo.model.parameters()) / 1024 / 1024

print("=" * 72)
print("  ESM-2 650M：原始模型 vs TurboESM  CUDA Benchmark")
print("=" * 72)
print(f"  模型参数显存  原始: {ref_param_mb:.1f} MB   Turbo: {turbo_param_mb:.1f} MB")
print(f"  warmup={WARMUP}, runs={RUNS}")
print()
print(f"{'序列':18} {'tok':>4}  "
      f"{'原始(ms)':>9} {'Turbo(ms)':>10} {'加速':>6}  "
      f"{'原始峰值MB':>10} {'Turbo峰值MB':>11} {'显存节省':>8}")
print("-" * 80)

for label, seq in test_sequences.items():
    inputs  = tokenizer(seq, return_tensors="pt").to(DEVICE)
    seq_len = inputs['input_ids'].shape[1]

    # 速度
    t_ref   = bench_ms(lambda: ref_model(**inputs))
    turbo.kv_cache.reset()
    t_turbo = bench_ms(lambda: (turbo.kv_cache.reset(), turbo.generate(seq)))

    # 峰值显存（激活值 + KV cache，不含模型参数）
    mem_ref   = measure_memory_mb(lambda: ref_model(**inputs))
    turbo.kv_cache.reset()
    mem_turbo = measure_memory_mb(lambda: (turbo.kv_cache.reset(), turbo.generate(seq)))

    speedup  = t_ref / t_turbo
    mem_save = (1 - mem_turbo / mem_ref) * 100 if mem_ref > 0 else 0

    print(f"{label:18} {seq_len:>4}  "
          f"{t_ref:>9.2f} {t_turbo:>10.2f} {speedup:>5.2f}x  "
          f"{mem_ref:>10.1f} {mem_turbo:>11.1f} {mem_save:>7.1f}%")

# ── KV Cache 显存分析 ─────────────────────────────────────────────────────────
print()
print("── KV Cache 显存理论分析 ──")
config     = turbo.config
num_layers = config.num_hidden_layers
num_heads  = config.num_attention_heads
head_dim   = config.hidden_size // num_heads
seq_len    = 512  # 假设最大序列长度

fp32_kv_mb  = 2 * num_layers * num_heads * seq_len * head_dim * 4 / 1024 / 1024
bit3_kv_mb  = 2 * num_layers * num_heads * seq_len * (head_dim / 8 * 4) / 1024 / 1024  # 3bit packed int32
qjl_mb      = num_layers * num_heads * seq_len * (head_dim / 32 * 4) / 1024 / 1024

print(f"  序列长度假设: {seq_len} tokens, {num_layers}层, {num_heads}头, head_dim={head_dim}")
print(f"  FP32 KV Cache:     {fp32_kv_mb:.1f} MB")
print(f"  3-bit packed K+V:  {bit3_kv_mb:.1f} MB")
print(f"  1-bit QJL signs:   {qjl_mb:.1f} MB")
print(f"  Turbo 总计:        {bit3_kv_mb + qjl_mb:.1f} MB")
print(f"  理论压缩比:        {fp32_kv_mb / (bit3_kv_mb + qjl_mb):.1f}x")

print()
print(f"  Triton 可用: {__import__('esm_turbo.triton_kernels', fromlist=['TRITON_AVAILABLE']).TRITON_AVAILABLE}")
