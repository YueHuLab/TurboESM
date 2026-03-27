import torch
import torch.nn as nn
from transformers.models.esm.modeling_esm import EsmSelfAttention
from .triton_kernels import TRITON_AVAILABLE, turbo_fused_decode_attention
import math

# ESM-2 RoPE 核心逻辑实现
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

class TurboEsmSelfAttention(EsmSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.register_buffer("pi_matrix", torch.eye(config.hidden_size // config.num_attention_heads, dtype=torch.float32).unsqueeze(0).repeat(config.num_attention_heads, 1, 1))
        self.register_buffer("lut_k", torch.zeros(config.num_attention_heads, 8, dtype=torch.float32))
        self.register_buffer("lut_v", torch.zeros(config.num_attention_heads, 8, dtype=torch.float32))
        # QJL residual scale：每个 head 的量化残差标准差，用于 decode 时修正
        self.register_buffer("residual_scale_k", torch.zeros(config.num_attention_heads, dtype=torch.float32))
        self.alpha = 0.5
        self.head_dim = config.hidden_size // config.num_attention_heads

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.transpose(1, 2)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        hidden_states = hidden_states.to(self.query.weight.dtype)
        bsz, seq_len, _ = hidden_states.size()

        # 从实例属性读取 kv_cache 和 layer_idx（由 TurboESM.generate 在调用前注入）
        turbo_kv_cache = getattr(self, '_turbo_kv_cache', None)
        layer_idx = getattr(self, '_layer_idx', None)
        
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))

        # ESM-2 在 RoPE 之前对 Q 预缩放（而非在 attention 内部缩放）
        q = q * (self.attention_head_size ** -0.5)

        # 获取当前位置索引（Decode 阶段需要正确的 start_pos）
        if seq_len == 1 and turbo_kv_cache is not None:
            start_pos = turbo_kv_cache.get_seq_len(layer_idx)
        else:
            start_pos = 0

        # 手动应用 RoPE
        if hasattr(self, "rotary_embeddings"):
            emb = self.rotary_embeddings
            # 若缓存不存在或长度不足当前序列，重新触发初始化
            cached_len = emb._cos_cached.shape[2] if emb._cos_cached is not None else 0
            if cached_len < start_pos + seq_len:
                emb(q, k)

            # 手动截取 cos/sin 表
            cos = emb._cos_cached[:, :, start_pos:start_pos+seq_len, :].to(q.dtype)
            sin = emb._sin_cached[:, :, start_pos:start_pos+seq_len, :].to(q.dtype)

            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)

        # 叠加 Pi 旋转：Q' = Q @ Pi^T，K' = K @ Pi^T
        # 内积等价：(Q Pi^T)(K Pi^T)^T = Q Pi^T Pi K^T = Q K^T
        pi_transposed = self.pi_matrix.transpose(-1, -2)
        q_rot = torch.einsum('bhsd,hde->bhse', q, pi_transposed)
        k_rot = torch.einsum('bhsd,hde->bhse', k, pi_transposed)

        if seq_len > 1:
            if turbo_kv_cache is not None:
                turbo_kv_cache.store_packed(layer_idx, k_rot, v, self.lut_k, self.lut_v)
            attn_output = torch.nn.functional.scaled_dot_product_attention(q_rot, k_rot, v, attn_mask=attention_mask, scale=1.0)
        else:
            if turbo_kv_cache is not None:
                turbo_kv_cache.store_packed(layer_idx, k_rot, v, self.lut_k, self.lut_v)
                if TRITON_AVAILABLE:
                    valid_len = turbo_kv_cache.layer_seq_len[layer_idx, 0].item()
                    k_packed = turbo_kv_cache.k_cache[layer_idx, :, :, :valid_len, :]
                    v_packed = turbo_kv_cache.v_cache[layer_idx, :, :, :valid_len, :]
                    qjl      = turbo_kv_cache.qjl_cache[layer_idx, :, :, :valid_len, :]
                    attn_output = turbo_fused_decode_attention(
                        q_rot, k_packed, v_packed, qjl,
                        self.lut_k, self.lut_v, self.residual_scale_k
                    )
                else:
                    k_full, v_full = turbo_kv_cache.fetch_unpacked(layer_idx, self.lut_k, self.lut_v, self.residual_scale_k)
                    attn_output = torch.nn.functional.scaled_dot_product_attention(q_rot, k_full, v_full, attn_mask=attention_mask, scale=1.0)
            else:
                attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, scale=1.0)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return attn_output, None

def inject_turbo_attention(model):
    for name, module in model.named_modules():
        if isinstance(module, EsmSelfAttention) and not isinstance(module, TurboEsmSelfAttention):
            parent_name = name.rsplit('.', 1)[0]
            parent_module = model.get_submodule(parent_name)
            turbo_attn = TurboEsmSelfAttention(model.config)
            turbo_attn.load_state_dict(module.state_dict(), strict=False)
            setattr(parent_module, "self", turbo_attn)
    return model
