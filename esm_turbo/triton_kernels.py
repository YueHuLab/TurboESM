# TurboQuant Triton CUDA Kernels
# Mac/MPS 环境自动跳过，CUDA 环境自动启用。
# 实现了两个核心 kernel：
#   1. turbo_dequant_k_kernel：3-bit 解包 + QJL 残差修正，融合为单个 kernel
#   2. turbo_decode_attention_kernel：完整的 fused decode attention（解包 + attention + 残差）

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def turbo_dequant_k_kernel(
        # 输入
        k_packed_ptr,    # (bsz, heads, seq, packed_dim) int32，3-bit 打包
        qjl_ptr,         # (bsz, heads, seq, qjl_dim)   int32，1-bit 打包残差符号
        lut_k_ptr,       # (heads, 8) float32，K 的量化码本
        residual_scale_ptr,  # (heads,) float32，每 head 的残差幅度
        # 输出
        k_out_ptr,       # (bsz, heads, seq, head_dim) float32
        # 尺寸
        bsz, num_heads, seq_len, head_dim, packed_dim, qjl_dim,
        # 常量
        BLOCK_SEQ: tl.constexpr,
        HEAD_DIM:  tl.constexpr,
    ):
        """
        融合 kernel：3-bit 解包 → LUT 反量化 → 1-bit QJL 残差修正 → 输出 FP32 K
        Grid: (bsz * num_heads, cdiv(seq_len, BLOCK_SEQ))
        """
        pid_bh  = tl.program_id(0)   # batch * head 索引
        pid_seq = tl.program_id(1)   # seq block 索引

        b_idx = pid_bh // num_heads
        h_idx = pid_bh % num_heads

        seq_start = pid_seq * BLOCK_SEQ
        seq_offs  = seq_start + tl.arange(0, BLOCK_SEQ)
        seq_mask  = seq_offs < seq_len

        # 加载 LUT（8 个量化点）
        lut_offs = h_idx * 8 + tl.arange(0, 8)
        lut = tl.load(lut_k_ptr + lut_offs)  # (8,)

        # 加载残差幅度
        res_scale = tl.load(residual_scale_ptr + h_idx)

        # 逐 packed word 解包（每个 int32 含 8 个 3-bit 值）
        dim_offs = tl.arange(0, HEAD_DIM)
        packed_col = dim_offs // 8
        bit_shift  = (dim_offs % 8) * 3

        packed_base = (b_idx * num_heads + h_idx) * seq_len * (HEAD_DIM // 8)

        for s in range(BLOCK_SEQ):
            seq_idx = seq_start + s
            if seq_idx < seq_len:
                # 加载该 token 的所有 packed words
                p_offs = packed_base + seq_idx * (HEAD_DIM // 8) + packed_col
                packed = tl.load(k_packed_ptr + p_offs)           # (HEAD_DIM,) int32

                # 解包：取 3-bit 索引
                indices = (packed >> bit_shift) & 0x7              # (HEAD_DIM,)

                # LUT 查表反量化
                k_vals = tl.gather(lut, indices, 0)                # (HEAD_DIM,)

                # QJL 残差修正：解包 1-bit 符号
                qjl_base = (b_idx * num_heads + h_idx) * seq_len * (HEAD_DIM // 32)
                qjl_col  = dim_offs // 32
                qjl_bit  = dim_offs % 32
                q_offs   = qjl_base + seq_idx * (HEAD_DIM // 32) + qjl_col
                qjl_packed = tl.load(qjl_ptr + q_offs)
                signs = ((qjl_packed >> qjl_bit) & 0x1).to(tl.float32) * 2.0 - 1.0

                k_vals = k_vals + signs * res_scale

                # 写出
                out_base = (b_idx * num_heads + h_idx) * seq_len * HEAD_DIM
                out_offs = out_base + seq_idx * HEAD_DIM + dim_offs
                tl.store(k_out_ptr + out_offs, k_vals)


    @triton.jit
    def turbo_decode_attention_kernel(
        # Q（已旋转，FP32）
        q_ptr,           # (bsz, heads, 1, head_dim)
        # K/V packed cache
        k_packed_ptr,    # (bsz, heads, seq, packed_dim) int32
        v_packed_ptr,    # (bsz, heads, seq, packed_dim) int32
        qjl_ptr,         # (bsz, heads, seq, qjl_dim)   int32
        # LUT 和残差
        lut_k_ptr,       # (heads, 8)
        lut_v_ptr,       # (heads, 8)
        residual_scale_ptr,  # (heads,)
        # 输出
        out_ptr,         # (bsz, heads, 1, head_dim)
        # 尺寸
        bsz, num_heads, seq_len, head_dim,
        # 常量
        BLOCK_KV:  tl.constexpr,
        HEAD_DIM:  tl.constexpr,
    ):
        """
        Fused Decode Attention kernel：
          1. 解包 3-bit K，叠加 QJL 残差
          2. 计算 Q·K^T scores（分块 softmax）
          3. 解包 3-bit V
          4. 加权求和输出
        Grid: (bsz * num_heads,)
        """
        pid = tl.program_id(0)
        b_idx = pid // num_heads
        h_idx = pid % num_heads

        # 加载 Q (1 × HEAD_DIM)
        q_base = (b_idx * num_heads + h_idx) * HEAD_DIM
        q = tl.load(q_ptr + q_base + tl.arange(0, HEAD_DIM))  # (HEAD_DIM,)

        # 加载 LUT
        lut_k = tl.load(lut_k_ptr + h_idx * 8 + tl.arange(0, 8))
        lut_v = tl.load(lut_v_ptr + h_idx * 8 + tl.arange(0, 8))
        res_scale = tl.load(residual_scale_ptr + h_idx)

        dim_offs   = tl.arange(0, HEAD_DIM)
        packed_col = dim_offs // 8
        bit_shift  = (dim_offs % 8) * 3
        qjl_col    = dim_offs // 32
        qjl_bit    = dim_offs % 32

        packed_dim = HEAD_DIM // 8
        qjl_dim    = HEAD_DIM // 32
        kv_base    = (b_idx * num_heads + h_idx) * seq_len

        # 分块 online softmax（Flash Attention 风格）
        m = tl.full((1,), float('-inf'), dtype=tl.float32)  # 当前最大值
        acc_denom = tl.zeros((1,), dtype=tl.float32)         # softmax 分母
        acc_num   = tl.zeros((HEAD_DIM,), dtype=tl.float32)  # 加权 V 累加

        for block_start in range(0, seq_len, BLOCK_KV):
            block_offs = block_start + tl.arange(0, BLOCK_KV)
            mask = block_offs < seq_len

            # 解包 K block
            k_base_block = kv_base * packed_dim
            k_p_offs = k_base_block + block_offs[:, None] * packed_dim + packed_col[None, :]
            k_packed  = tl.load(k_packed_ptr + k_p_offs, mask=mask[:, None], other=0)
            k_indices = (k_packed >> bit_shift[None, :]) & 0x7
            k_vals    = tl.gather(lut_k, k_indices.reshape(BLOCK_KV * HEAD_DIM), 0).reshape(BLOCK_KV, HEAD_DIM)

            # QJL 残差修正
            qjl_base_block = kv_base * qjl_dim
            q_p_offs  = qjl_base_block + block_offs[:, None] * qjl_dim + qjl_col[None, :]
            qjl_p     = tl.load(qjl_ptr + q_p_offs, mask=mask[:, None], other=0)
            signs     = (((qjl_p >> qjl_bit[None, :]) & 0x1).to(tl.float32)) * 2.0 - 1.0
            k_vals    = k_vals + signs * res_scale

            # Q·K^T scores: (BLOCK_KV,)
            scores = tl.sum(q[None, :] * k_vals, axis=1)  # (BLOCK_KV,)
            scores = tl.where(mask, scores, float('-inf'))

            # Online softmax 更新
            m_new  = tl.maximum(m, tl.max(scores, axis=0))
            exp_s  = tl.exp(scores - m_new)
            scale  = tl.exp(m - m_new)
            acc_denom = acc_denom * scale + tl.sum(exp_s, axis=0)

            # 解包 V block
            v_base_block = kv_base * packed_dim
            v_p_offs  = v_base_block + block_offs[:, None] * packed_dim + packed_col[None, :]
            v_packed  = tl.load(v_packed_ptr + v_p_offs, mask=mask[:, None], other=0)
            v_indices = (v_packed >> bit_shift[None, :]) & 0x7
            v_vals    = tl.gather(lut_v, v_indices.reshape(BLOCK_KV * HEAD_DIM), 0).reshape(BLOCK_KV, HEAD_DIM)

            # 加权 V 累加
            acc_num = acc_num * scale + tl.sum(exp_s[:, None] * v_vals, axis=0)
            m = m_new

        # 归一化输出
        out = acc_num / acc_denom
        out_base = (b_idx * num_heads + h_idx) * HEAD_DIM
        tl.store(out_ptr + out_base + dim_offs, out)


    def turbo_dequant_k(k_packed, qjl, lut_k, residual_scale, head_dim):
        """Python 包装：调用 turbo_dequant_k_kernel，返回 FP32 K tensor。"""
        import torch
        bsz, num_heads, seq_len, packed_dim = k_packed.shape
        k_out = torch.empty(bsz, num_heads, seq_len, head_dim,
                            dtype=torch.float32, device=k_packed.device)
        BLOCK_SEQ = 16
        grid = (bsz * num_heads, triton.cdiv(seq_len, BLOCK_SEQ))
        turbo_dequant_k_kernel[grid](
            k_packed.contiguous(), qjl.contiguous(), lut_k, residual_scale, k_out,
            bsz, num_heads, seq_len, head_dim, packed_dim, head_dim // 32,
            BLOCK_SEQ=BLOCK_SEQ, HEAD_DIM=head_dim,
        )
        return k_out


    def turbo_fused_decode_attention(q, k_packed, v_packed, qjl,
                                      lut_k, lut_v, residual_scale):
        """
        Python 包装：fused decode attention。
        q: (bsz, heads, 1, head_dim) FP32，已做 Pi 旋转和预缩放
        返回: (bsz, heads, 1, head_dim) FP32
        """
        import torch
        bsz, num_heads, _, head_dim = q.shape
        seq_len = k_packed.shape[2]
        out = torch.empty(bsz, num_heads, head_dim,
                          dtype=torch.float32, device=q.device)
        BLOCK_KV = min(64, triton.next_power_of_2(seq_len))
        grid = (bsz * num_heads,)
        turbo_decode_attention_kernel[grid](
            q.contiguous(), k_packed.contiguous(), v_packed.contiguous(), qjl.contiguous(),
            lut_k, lut_v, residual_scale, out,
            bsz, num_heads, seq_len, head_dim,
            BLOCK_KV=BLOCK_KV, HEAD_DIM=head_dim,
        )
        return out.unsqueeze(2)  # (bsz, heads, 1, head_dim)

else:
    # Mac/CPU 降级：给出明确提示而非静默失败
    def turbo_dequant_k(*args, **kwargs):
        raise RuntimeError(
            "Triton 不可用（当前环境为 Mac/CPU）。"
            "请在 CUDA 环境下安装 triton：pip install triton"
        )

    def turbo_fused_decode_attention(*args, **kwargs):
        raise RuntimeError(
            "Triton 不可用（当前环境为 Mac/CPU）。"
            "请在 CUDA 环境下安装 triton：pip install triton"
        )
