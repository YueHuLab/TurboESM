"""
Colab 精度对齐验证脚本：对比 TurboESM 与原始 ESM-2 650M 的 cosine 相似度。
依赖 colab_setup.py 已执行（turbo / MODEL_LOCAL / DEVICE 已定义）。
"""
import glob, shutil, sys, torch, torch.nn.functional as F
from transformers import EsmModel, EsmTokenizer

# ── 环境初始化（与 colab_setup.py 一致）────────────────────────────────────
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

# ── 加载模型 ────────────────────────────────────────────────────────────────
tokenizer = EsmTokenizer.from_pretrained(MODEL_LOCAL)
ref_model = EsmModel.from_pretrained(MODEL_LOCAL).to(DEVICE).eval()
turbo     = TurboESM(model_dir=MODEL_LOCAL, checkpoint_path=CKPT_PATH, device=DEVICE)

# ── 测试序列 ────────────────────────────────────────────────────────────────
test_sequences = {
    "短肽-胰岛素B链":    "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
    "中等-血红蛋白α":    "VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
    "疏水-跨膜区":       "MALLLLGFALLAGTAMAFGFGFGFGFGFGFGFGFGFG",
    "重复序列-低复杂度":  "MAPLRKTYLLPVLLGLLAAAPAPAPAPAPAPAPAPA",
    "酶-DHFR片段":       "GASTEFLSYYVDQINTFNLTTPRQRLIVDRGEKIGYYTPVKLDAGMKEF",
    "IDR-低复杂度长序列": "MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQSSSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNS",
}

# ── 精度对比 ────────────────────────────────────────────────────────────────
print(f"\n{'序列名':<22} {'长度':>4}  {'Prefill 相似度':>14}  {'结论':>6}")
print("-" * 55)

all_pass = True
for name, seq in test_sequences.items():
    inputs  = tokenizer(seq, return_tensors="pt").to(DEVICE)
    seq_len = inputs['input_ids'].shape[1]

    with torch.no_grad():
        ref_out   = ref_model(**inputs).last_hidden_state
        turbo.kv_cache.reset()
        turbo_out = turbo.generate(seq)

    sim = F.cosine_similarity(ref_out[0].cpu(), turbo_out[0].cpu(), dim=-1).mean().item()
    ok  = sim > 0.95
    if not ok:
        all_pass = False
    print(f"{name:<22} {seq_len:>4}  {sim:>14.6f}  {'✅' if ok else '⚠️'}")

print("-" * 55)
if all_pass:
    print("✅ 全部序列相似度 > 0.95，TurboQuant 精度验证通过！")
else:
    print("⚠️  部分序列未达标，请检查校准权重。")
