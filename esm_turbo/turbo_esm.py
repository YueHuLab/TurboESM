import torch
from transformers import EsmModel, EsmConfig, EsmTokenizer
from .modeling_esm_turbo import inject_turbo_attention
from .kv_cache import TurboKVCache
import os

class TurboESM:
    """
    极简统一入口：将 ESM2 封装为 Turbo 加速版本
    """
    def __init__(self, model_dir=None, checkpoint_path=None, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
            
        # 获取当前文件所在目录，从而定位 weights
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sdk_root = os.path.dirname(current_dir)
        
        if model_dir is None:
            model_dir = os.path.join(sdk_root, "weights/esm2_650M")
        if checkpoint_path is None:
            checkpoint_path = os.path.join(sdk_root, "weights/esm2_650M_turbo.pt")
            
        model_dir = os.path.abspath(model_dir)
        checkpoint_path = os.path.abspath(checkpoint_path)
        
        print(f"[*] 初始化 TurboESM 引擎 (设备: {self.device})...")
        print(f"[*] 加载路径: {model_dir}")
        
        # 1. 加载配置和基础模型
        self.config = EsmConfig.from_pretrained(model_dir)
        self.model = EsmModel(self.config)
        
        # 2. 核心手术：注入 TurboAttention 算子
        self.model = inject_turbo_attention(self.model)
        
        # 3. 加载 TurboQuant 专用权重
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"找不到 Turbo 权重文件: {checkpoint_path}。请先运行 python builder.py 生成。")
            
        state_dict = torch.load(checkpoint_path, weights_only=True, map_location=self.device)
        # 确保 pi_matrix/lut 与模型主体同为 f32，避免 MPS 混精度 crash
        for k in state_dict:
            if k.endswith('pi_matrix') or k.endswith('lut_k') or k.endswith('lut_v') or k.endswith('residual_scale_k'):
                state_dict[k] = state_dict[k].float()
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        # 4. 初始化 Tokenizer
        self.tokenizer = EsmTokenizer.from_pretrained(model_dir)
        
        # 5. 预分配专用显存池 (KV Cache)
        self.kv_cache = TurboKVCache(self.config, max_batch_size=1, max_seq_len=1024, device=self.device)
        print("[+] TurboESM 准备就绪，算子已深度融合。")

    @torch.no_grad()
    def generate(self, sequence):
        """
        加速推理演示
        """
        inputs = self.tokenizer(sequence, return_tensors="pt").to(self.device)

        # 将 kv_cache 绑定到每个 TurboEsmSelfAttention 层上，由 forward 内部读取
        # 这样绕过了 EsmModel.forward 不透传自定义参数的限制
        for layer_idx, layer in enumerate(self.model.encoder.layer):
            attn = layer.attention.self
            attn._turbo_kv_cache = self.kv_cache
            attn._layer_idx = layer_idx

        outputs = self.model(**inputs)

        return outputs.last_hidden_state

if __name__ == "__main__":
    # 使用起来就像原生模型一样简单
    esm_turbo = TurboESM()
    protein_seq = "MAPLRKTYLLPV"
    result = esm_turbo.generate(protein_seq)
    print(f"推理成功，输出特征维度: {result.shape}")
