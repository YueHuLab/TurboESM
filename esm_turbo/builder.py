import torch
import scipy.cluster.vq as vq # 用于 K-means 聚类
from transformers import EsmModel
import os

def generate_turbo_weights(model_path="./esm2_650M", save_path="esm2_650M_turbo.pt"):
    print(f"Loading original ESM model from: {model_path}")
    model = EsmModel.from_pretrained(model_path)
    state_dict = model.state_dict()
    
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    turbo_state_dict = {}
    
    print("Starting generation of TurboQuant pre-computed matrices...")
    for layer in range(num_layers):
        # 1. 为每一层的每个 Head 生成正交旋转矩阵 Pi
        # 使用随机高斯矩阵的 QR 分解来获得均匀分布的正交矩阵
        # (num_heads, head_dim, head_dim)
        random_matrix = torch.randn(num_heads, head_dim, head_dim)
        Q, R = torch.linalg.qr(random_matrix)
        # 确保行列式为 1 (可选，保证纯旋转无翻转)
        D = torch.diagonal(R, dim1=-2, dim2=-1)
        Pi = Q * torch.sign(D).unsqueeze(-2) 
        
        turbo_state_dict[f"encoder.layer.{layer}.attention.self.pi_matrix"] = Pi.half()
        
        # 2. 计算 Lloyd-Max 聚类中心 (LUT) - 这里使用数学先验
        # 根据高维超球面投影定理，旋转后服从 N(0, 1/d)
        # 我们用 K-means 生成 8 个最优量化点 (3-bit)
        simulated_data = torch.randn(100000) * (1.0 / (head_dim ** 0.5))
        centroids, _ = vq.kmeans(simulated_data.numpy(), 8)
        centroids = torch.tensor(centroids).sort()[0] # 排序保证索引对应大小
        
        # 将相同的 LUT 广播给所有 Head (也可以按 Head 独立生成)
        lut = centroids.repeat(num_heads, 1)
        turbo_state_dict[f"encoder.layer.{layer}.attention.self.lut"] = lut.half()
        
    print("Merging original weights with Turbo weights...")
    state_dict.update(turbo_state_dict)
    
    torch.save(state_dict, save_path)
    print(f"TurboQuant Checkpoint saved to: {save_path}")

if __name__ == "__main__":
    generate_turbo_weights()
