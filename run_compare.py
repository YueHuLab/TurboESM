"""
扩大测试序列，验证 TurboQuant 在不同长度、不同结构类型蛋白质上的准确性。
"""
import torch
import torch.nn.functional as F
from transformers import EsmModel, EsmTokenizer
from esm_turbo.turbo_esm import TurboESM

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
MODEL_DIR = 'weights/esm2_650M'
CKPT = 'weights/esm2_650M_turbo.pt'

test_sequences = {
    "短肽-胰岛素B链":       "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
    "中等-血红蛋白α":       "VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
    "疏水-跨膜区":          "MALLLLGFALLAGTAMAFGFGFGFGFGFGFGFGFGFG",
    "重复序列-低复杂度":     "MAPLRKTYLLPVLLGLLAAAPAPAPAPAPAPAPAPA",
    "酶-DHFR片段":          "GASTEFLSYYVDQINTFNLTTPRQRLIVDRGEKIGYYTPVKLDAGMKEF",
    "IDR-低复杂度长序列":    "MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQSSSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNS",
}

tokenizer = EsmTokenizer.from_pretrained(MODEL_DIR)
orig = EsmModel.from_pretrained(MODEL_DIR).to(device).eval()
turbo = TurboESM(model_dir=MODEL_DIR, checkpoint_path=CKPT, device=device)

print(f"\n{'序列名':<22} {'长度':>4}  {'Prefill 相似度':>14}  {'Decode 相似度(L0)':>18}")
print("-" * 68)

for name, seq in test_sequences.items():
    inputs = tokenizer(seq, return_tensors='pt').to(device)
    seq_len = inputs['input_ids'].shape[1]

    # 原始模型基准
    with torch.no_grad():
        ref = orig(**inputs).last_hidden_state

    # Prefill
    turbo.kv_cache.reset()
    out = turbo.generate(seq)
    sim_prefill = F.cosine_similarity(ref.float(), out.float(), dim=-1).mean().item()

    # Decode：用 layer 0 的量化 KV cache 做 attention，对比原始
    captured = {}
    def hook(module, inp, outp):
        hs = inp[0]
        bsz, sl, _ = hs.shape
        nh = turbo.config.num_attention_heads
        hd = turbo.config.hidden_size // nh
        with torch.no_grad():
            q = module.query(hs).view(bsz, sl, nh, hd).transpose(1, 2)
            k = module.key(hs).view(bsz, sl, nh, hd).transpose(1, 2)
            v = module.value(hs).view(bsz, sl, nh, hd).transpose(1, 2)
            q = q * (hd ** -0.5)
            q, k = module.rotary_embeddings(q, k)
        captured['q'] = q.detach().cpu()
        captured['k'] = k.detach().cpu()
        captured['v'] = v.detach().cpu()

    h = orig.encoder.layer[0].attention.self.register_forward_hook(hook)
    with torch.no_grad():
        orig(**inputs)
    h.remove()

    q_orig = captured['q']
    k_orig = captured['k']
    v_orig = captured['v']

    pi0    = turbo.model.encoder.layer[0].attention.self.pi_matrix
    lut0_k = turbo.model.encoder.layer[0].attention.self.lut_k
    lut0_v = turbo.model.encoder.layer[0].attention.self.lut_v
    pi_t   = pi0.transpose(-1, -2)

    turbo.kv_cache.reset()
    _ = turbo.generate(seq)
    k_full, v_full = turbo.kv_cache.fetch_unpacked(0, lut0_k, lut0_v)

    # 把 CPU 上的 q/k/v 移到正确设备做计算
    q_orig_d = q_orig.to(device)
    k_orig_d = k_orig.to(device)
    v_orig_d = v_orig.to(device)

    q_rot = torch.einsum('bhsd,hde->bhse', q_orig_d, pi_t)
    ref_attn   = F.scaled_dot_product_attention(q_rot, torch.einsum('bhsd,hde->bhse', k_orig_d, pi_t), v_orig_d, scale=1.0)
    turbo_attn = F.scaled_dot_product_attention(q_rot, k_full, v_full, scale=1.0)
    sim_decode = F.cosine_similarity(ref_attn.float(), turbo_attn.float(), dim=-1).mean().item()

    status_p = "✓" if sim_prefill >= 0.99 else "✗"
    status_d = "✓" if sim_decode  >= 0.95 else "✗"
    print(f"{name:<22} {seq_len:>4}  {sim_prefill:>12.4f} {status_p}  {sim_decode:>16.4f} {status_d}")

print("\n目标: Prefill >0.99, Decode >0.95")
