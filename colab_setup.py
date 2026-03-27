"""
Colab 初始化脚本：清除 pyc 缓存，定位本地 HuggingFace 模型缓存，加载 TurboESM。
每次 Colab Runtime 重启后，或修改过 esm_turbo/ 下任何文件后，运行此脚本。
"""
import glob, shutil, sys, torch, torch.nn.functional as F
from transformers import EsmModel, EsmTokenizer

# ── 1. 清除 pyc 缓存，强制重新加载最新代码 ──────────────────────────────────
shutil.rmtree('/content/esm_turbo/__pycache__', ignore_errors=True)
for k in [k for k in sys.modules if 'esm_turbo' in k]:
    del sys.modules[k]

# ── 2. 定位本地 HuggingFace 缓存（避免 abspath 把 HF ID 变成本地路径）────────
candidates = glob.glob(
    '/root/.cache/huggingface/hub/models--facebook--esm2_t33_650M_UR50D/snapshots/*/config.json'
)
if not candidates:
    raise FileNotFoundError(
        "未找到本地 ESM-2 缓存，请先运行：\n"
        "  from transformers import EsmModel; EsmModel.from_pretrained('facebook/esm2_t33_650M_UR50D')"
    )
MODEL_LOCAL = candidates[0].replace('/config.json', '')
print(f"[*] 使用本地模型缓存: {MODEL_LOCAL}")

CKPT_PATH = '/content/weights/esm2_650M_turbo.pt'
DEVICE    = 'cuda'

# ── 3. 加载 TurboESM ────────────────────────────────────────────────────────
from esm_turbo.turbo_esm import TurboESM
turbo    = TurboESM(model_dir=MODEL_LOCAL, checkpoint_path=CKPT_PATH, device=DEVICE)
tokenizer = EsmModel.from_pretrained  # 仅占位，turbo 自带 tokenizer

print(f"[+] turbo.model 类型: {type(turbo.model).__name__}")
print("[+] TurboESM 初始化完成，可直接调用 turbo.generate(seq)")
