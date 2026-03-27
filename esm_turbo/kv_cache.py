import torch
import torch.nn as nn
from .triton_kernels import TRITON_AVAILABLE, turbo_dequant_k

class TurboKVCache:
    """
    A custom memory manager for 3-bit packed KV cache.
    Uses INT32 tensors to store 3-bit quantized K and V values (8 elements per INT32).
    Also stores 1-bit QJL signs (32 elements per INT32).
    """
    def __init__(self, config, max_batch_size, max_seq_len, device="cuda"):
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.device = device
        
        # 8 elements per INT32 (3-bit each = 24 bits used)
        self.packed_dim = self.head_dim // 8
        
        # 32 elements per INT32 (1-bit each = 32 bits used)
        self.qjl_packed_dim = self.head_dim // 32
        
        # KV Cache for K (3-bit packed)
        self.k_cache = torch.zeros(
            (self.num_layers, max_batch_size, self.num_heads, max_seq_len, self.packed_dim),
            dtype=torch.int32, device=device
        )
        
        # KV Cache for V (3-bit packed)
        self.v_cache = torch.zeros(
            (self.num_layers, max_batch_size, self.num_heads, max_seq_len, self.packed_dim),
            dtype=torch.int32, device=device
        )
        
        # QJL signs cache (1-bit packed)
        self.qjl_cache = torch.zeros(
            (self.num_layers, max_batch_size, self.num_heads, max_seq_len, self.qjl_packed_dim),
            dtype=torch.int32, device=device
        )
        
        # 每层独立记录已存长度，避免 Decode 时 position 错位
        # curr_seq_len 在最后一层写完后更新，其余层通过 start_idx 正确定位
        self.layer_seq_len = torch.zeros(self.num_layers, max_batch_size, dtype=torch.int32, device=device)

        # Current length of KV cache per batch (以最后一层为准，供外部读取)
        self.curr_seq_len = torch.zeros(max_batch_size, dtype=torch.int32, device=device)

    def store_packed(self, layer_idx, k_rotated, v_rotated, lut_k, lut_v):
        """
        Store rotated K and V after quantization and packing.
        K and V use independent LUTs for optimal quantization accuracy.
        QJL signs are computed internally from the quantization residual.
        """
        bsz, num_heads, seq_len, head_dim = k_rotated.size()
        start_idx = self.layer_seq_len[layer_idx, 0].item()

        # 用 bucketize 一步完成量化+sign，比 argmin+gather 快
        k_quant, signs = self._quantize_with_signs(k_rotated, lut_k)
        v_quant = self._quantize(v_rotated, lut_v)

        k_packed = self._pack_3bit(k_quant)
        v_packed = self._pack_3bit(v_quant)
        qjl_packed = self._pack_1bit(signs)

        end_idx = start_idx + seq_len
        self.k_cache[layer_idx, :bsz, :, start_idx:end_idx, :] = k_packed
        self.v_cache[layer_idx, :bsz, :, start_idx:end_idx, :] = v_packed
        self.qjl_cache[layer_idx, :bsz, :, start_idx:end_idx, :] = qjl_packed

        self.layer_seq_len[layer_idx, :bsz] += seq_len
        if layer_idx == self.num_layers - 1:
            self.curr_seq_len[:bsz] = self.layer_seq_len[layer_idx, :bsz]

    def fetch_unpacked(self, layer_idx, lut_k, lut_v, residual_scale_k=None):
        """
        将 3-bit 压缩数据解压回 FP32 张量，仅解压已填充的部分。
        CUDA 环境：自动走 Triton kernel（融合解包 + QJL 修正）。
        Mac/CPU 环境：纯 PyTorch 实现。
        """
        valid_len = self.layer_seq_len[layer_idx, 0].item()
        k_packed = self.k_cache[layer_idx, :, :, :valid_len, :]
        v_packed = self.v_cache[layer_idx, :, :, :valid_len, :]

        # CUDA 路径：Triton kernel 融合解包 + QJL 修正
        if TRITON_AVAILABLE and k_packed.is_cuda and residual_scale_k is not None:
            qjl = self.qjl_cache[layer_idx, :, :, :valid_len, :]
            k_dequant = turbo_dequant_k(
                k_packed, qjl, lut_k, residual_scale_k, self.head_dim
            )
            # V 仍走 PyTorch（V 无 QJL 修正）
            v_unquant = self._unpack_3bit(v_packed)
            bsz, num_heads, seq_len, _ = v_unquant.size()
            lut_v_exp = lut_v.to(v_unquant.device).view(1, num_heads, 1, 8).expand(bsz, -1, seq_len, -1)
            v_dequant = torch.gather(lut_v_exp, -1, v_unquant.long()).to(torch.float32)
            return k_dequant, v_dequant

        # Mac/CPU 路径：纯 PyTorch
        k_unquant = self._unpack_3bit(k_packed)
        v_unquant = self._unpack_3bit(v_packed)

        bsz, num_heads, seq_len, head_dim = k_unquant.size()
        dev = k_unquant.device
        lut_k_exp = lut_k.to(dev).view(1, num_heads, 1, 8).expand(bsz, -1, seq_len, -1)
        lut_v_exp = lut_v.to(dev).view(1, num_heads, 1, 8).expand(bsz, -1, seq_len, -1)
        k_dequant = torch.gather(lut_k_exp, -1, k_unquant.long())
        v_dequant = torch.gather(lut_v_exp, -1, v_unquant.long())

        # QJL residual 修正
        if residual_scale_k is not None:
            signs_packed = self.qjl_cache[layer_idx, :, :, :valid_len, :]
            signs = self._unpack_1bit(signs_packed)
            signs_pm = signs.float() * 2 - 1
            scale = residual_scale_k.to(dev).view(1, num_heads, 1, 1)
            k_dequant = k_dequant + signs_pm * scale

        return k_dequant.to(torch.float32), v_dequant.to(torch.float32)

    def _quantize_k(self, layer_idx, k_rotated, lut_k):
        """返回量化后的 K 值（用于计算残差符号），不存 cache。"""
        bsz, num_heads, seq_len, head_dim = k_rotated.size()
        x_flat = k_rotated.reshape(bsz, num_heads, -1)
        lut_expanded = lut_k.to(x_flat.device).unsqueeze(1)
        dist = torch.abs(x_flat.unsqueeze(-1) - lut_expanded)
        indices = torch.argmin(dist, dim=-1)
        lut_exp = lut_k.to(x_flat.device)[None, :, None, :].expand(bsz, -1, seq_len * head_dim, -1)
        quant_flat = torch.gather(lut_exp, -1, indices.unsqueeze(-1)).squeeze(-1)
        return quant_flat.view(bsz, num_heads, seq_len, head_dim)

    def _quantize_with_signs(self, x, lut):
        """一次 argmin 同时完成量化和残差符号计算，比两次调用省一次 dist 计算。"""
        bsz, num_heads, seq_len, head_dim = x.size()
        x_flat = x.reshape(bsz, num_heads, -1)
        lut_exp = lut.unsqueeze(0).unsqueeze(2)           # (1, heads, 1, 8)
        dist = torch.abs(x_flat.unsqueeze(-1) - lut_exp)  # (bsz, heads, seq*dim, 8)
        indices = torch.argmin(dist, dim=-1)               # (bsz, heads, seq*dim)
        quant_flat = torch.gather(lut_exp.expand(bsz, -1, seq_len * head_dim, -1),
                                  -1, indices.unsqueeze(-1)).squeeze(-1)
        signs = torch.sign(x_flat - quant_flat).view(bsz, num_heads, seq_len, head_dim)
        return indices.view(bsz, num_heads, seq_len, head_dim).to(torch.int32), signs

    def _quantize(self, x, lut):
        bsz, num_heads, seq_len, head_dim = x.size()
        x_flat = x.reshape(bsz, num_heads, -1)
        dist = torch.abs(x_flat.unsqueeze(-1) - lut.unsqueeze(0).unsqueeze(2))
        indices = torch.argmin(dist, dim=-1)
        return indices.view(bsz, num_heads, seq_len, head_dim).to(torch.int32)

    def _pack_3bit(self, x_quant):
        bsz, num_heads, seq_len, head_dim = x_quant.size()
        x_reshaped = x_quant.reshape(bsz, num_heads, seq_len, -1, 8)  # (..., groups, 8)
        # 向量化：shifts shape (8,)，一次广播完成所有位移
        shifts = torch.arange(0, 24, 3, dtype=torch.int32, device=x_quant.device)
        packed = (x_reshaped << shifts).sum(dim=-1).to(torch.int32)
        return packed

    def _unpack_3bit(self, packed):
        bsz, num_heads, seq_len, packed_dim = packed.size()
        # 向量化：shifts shape (8,)
        shifts = torch.arange(0, 24, 3, dtype=torch.int32, device=packed.device)
        unpacked = (packed.unsqueeze(-1) >> shifts) & 0x7  # (..., packed_dim, 8)
        return unpacked.reshape(bsz, num_heads, seq_len, -1)

    def _pack_1bit(self, signs):
        bits = (signs > 0).to(torch.int32)
        bsz, num_heads, seq_len, head_dim = bits.size()
        bits_reshaped = bits.reshape(bsz, num_heads, seq_len, -1, 32)  # (..., groups, 32)
        shifts = torch.arange(32, dtype=torch.int32, device=bits.device)
        packed = (bits_reshaped << shifts).sum(dim=-1).to(torch.int32)
        return packed

    def _unpack_1bit(self, packed):
        bsz, num_heads, seq_len, packed_dim = packed.size()
        unpacked = torch.zeros((bsz, num_heads, seq_len, packed_dim * 32), dtype=torch.int32, device=packed.device)
        for i in range(32):
            unpacked[..., i::32] = (packed >> i) & 0x1
        return unpacked

    def get_k_ptr(self, layer_idx):
        return self.k_cache[layer_idx].data_ptr()

    def get_v_ptr(self, layer_idx):
        return self.v_cache[layer_idx].data_ptr()

    def get_qjl_ptr(self, layer_idx):
        return self.qjl_cache[layer_idx].data_ptr()

    def get_seq_len(self, layer_idx=None):
        if layer_idx is not None:
            return self.layer_seq_len[layer_idx, 0].item()
        return self.curr_seq_len[0].item()

    def reset(self):
        self.curr_seq_len.zero_()
        self.layer_seq_len.zero_()
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.qjl_cache.zero_()
