import torch
import scipy.cluster.vq as vq
from transformers import EsmModel, EsmTokenizer
import os

def calibrate_turbo_weights(model_path="./esm2_650M", save_path="esm2_650M_turbo.pt"):
    print(f"[*] 开始【深度校准】TurboQuant 权重 (路径: {model_path})...")
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    model = EsmModel.from_pretrained(model_path).to(device).eval()
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    
    calibration_sequences = [
        # 短肽 / 功能域
        "MKTVRQERLKSIVVLGAGGVGSAVADYLRQKGIPVT",
        "MAPLRKTYLLPVLLGLLAAAPAPAPAPAPAPAPAPA",
        "GIVEQCCTSICSLYQLENYCN",
        "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
        # 酶活性位点附近序列
        "GASTEFLSYYVDQINTFNLTTPRQRLIVDRGEKIGYYTPVKLDAGMKEF",
        "KVFERCELARTLKRLGMDGYRGISLANWMCLAKWESGYNTRATNYNAGDRSTDYGIFQINSRYWCNDGKTPGAVNACHLSCSALLQDNIADAVACAKRVVRDPQGIRAWVAWRNRCQNRDVRQYVQGCGV",
        # 膜蛋白跨膜区（疏水性强）
        "MALLLLGFALLAGTAMAFGFGFGFGFGFGFGFGFGFG",
        "MSSNDQQGQFAKPGDTSSSSSSSSSSSSSSSSSSSSSS",
        # 球蛋白结构
        "VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
        # IDR（本征无序区，多样残基）
        "MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQSSSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNS",
    ]
    
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    # 使用 Hooks 捕获 RoPE 后的 K 和原始 V
    print("[1/3] 收集 RoPE 激活数据 (通过 Hooks)...")
    raw_k_data = {l: [] for l in range(num_layers)}
    raw_v_data = {l: [] for l in range(num_layers)}

    # 临时存储 K 和 V 的函数
    def get_hook(layer_idx):
        def hook(module, input, output):
            hidden_states = input[0]
            with torch.no_grad():
                key_states = module.key(hidden_states)
                query_states = module.query(hidden_states)
                value_states = module.value(hidden_states)

                bsz, seq_len, _ = key_states.shape
                q = query_states.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
                k = key_states.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
                v = value_states.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)

                if hasattr(module, "rotary_embeddings"):
                    _, k = module.rotary_embeddings(q, k)

                raw_k_data[layer_idx].append(k.cpu().float())
                raw_v_data[layer_idx].append(v.cpu().float())
        return hook

    hooks = []
    for l in range(num_layers):
        h = model.encoder.layer[l].attention.self.register_forward_hook(get_hook(l))
        hooks.append(h)

    # 运行前向传播
    with torch.no_grad():
        for seq in calibration_sequences:
            inputs = tokenizer(seq, return_tensors="pt").to(device)
            model(**inputs)

    # 移除 Hooks
    for h in hooks:
        h.remove()

    print("[2/3] 执行 SVD 正交化与 K-means 优化...")
    turbo_state_dict = {}
    original_state_dict = model.state_dict()
    
    for l in range(num_layers):
        layer_k_all = torch.cat(raw_k_data[l], dim=2)
        layer_v_all = torch.cat(raw_v_data[l], dim=2)
        layer_pi = []
        layer_lut_k = []
        layer_lut_v = []
        layer_residual_scale_k = []

        for h in range(num_heads):
            head_k = layer_k_all[0, h, :, :]
            head_v = layer_v_all[0, h, :, :]
            try:
                _, _, Vh = torch.linalg.svd(head_k.float(), full_matrices=False)
                Pi = Vh
            except:
                Pi = torch.eye(head_dim)

            layer_pi.append(Pi.unsqueeze(0))

            # K LUT：在旋转后的 K 上做 K-means
            head_rotated = (head_k @ Pi.T).float()
            centroids_k, _ = vq.kmeans(head_rotated.flatten().numpy(), 8)
            centroids_k = torch.tensor(centroids_k).sort()[0]
            layer_lut_k.append(centroids_k.unsqueeze(0))

            # 计算量化残差标准差，作为 QJL residual scale
            flat = head_rotated.flatten()
            dist = (flat.unsqueeze(-1) - centroids_k).abs()
            idx = dist.argmin(dim=-1)
            k_quant_vals = centroids_k[idx]
            residual = flat - k_quant_vals
            layer_residual_scale_k.append(residual.abs().mean().unsqueeze(0))

            # V LUT：直接在原始 V 上做 K-means
            centroids_v, _ = vq.kmeans(head_v.float().flatten().numpy(), 8)
            centroids_v = torch.tensor(centroids_v).sort()[0]
            layer_lut_v.append(centroids_v.unsqueeze(0))

        turbo_state_dict[f"encoder.layer.{l}.attention.self.pi_matrix"] = torch.cat(layer_pi).float()
        turbo_state_dict[f"encoder.layer.{l}.attention.self.lut_k"] = torch.cat(layer_lut_k).float()
        turbo_state_dict[f"encoder.layer.{l}.attention.self.lut_v"] = torch.cat(layer_lut_v).float()
        turbo_state_dict[f"encoder.layer.{l}.attention.self.residual_scale_k"] = torch.cat(layer_residual_scale_k).float()

    print(f"[3/3] 正在保存检查点...")
    original_state_dict.update(turbo_state_dict)
    torch.save(original_state_dict, save_path)
    print(f"[+] 深度校准完成！")

if __name__ == "__main__":
    calibrate_turbo_weights()
