import torch
import torch.nn as nn

class LEPC_NET(nn.Module):
   
    def __init__(self, feature_dim: int, hidden_dim_ratio: float = 0.5):
        """
        初始化模块。

        参数:
            feature_dim (int): 输入和输出特征的维度 (D)。
                               根据您的描述，这里是 512。
            hidden_dim_ratio (float): 门控网络中隐藏层维度相对于特征维度的比例。
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # --- 门控网络 (Gating MLP) ---
        # 它的输入是 F_aligned 和 F_residual 的拼接，所以输入维度是 2*D。
        # 输出是与特征维度相同的门控权重 g，所以输出维度是 D。
        gate_input_dim = 2 * feature_dim
        gate_hidden_dim = int(feature_dim * hidden_dim_ratio)
        gate_output_dim = feature_dim
        
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, gate_output_dim),
            nn.Sigmoid()  # 确保门控权重 g 的值在 (0, 1) 之间
        )
        
        # --- 层归一化 (Layer Normalization) ---
        # 对最终重组的特征进行归一化，以稳定训练。
        self.layer_norm = nn.LayerNorm(self.feature_dim)

    def forward(self, f_visual: torch.Tensor, p_text: torch.Tensor) -> torch.Tensor:
        """
        模块的前向传播逻辑。

        参数:
            f_visual (torch.Tensor): 批次的视觉特征，形状为 (B, D)。
            p_text (torch.Tensor): 批次的文本原型，形状为 (B, D)。

        返回:
            torch.Tensor: 经过解构与重组后的语义增强特征，形状为 (B, D)。
        """
        # --- 步骤 1: 语义投影 (Semantic Projection) ---
        # 计算 f_visual 在 p_text 上的投影，得到 F_aligned。
        # 公式: proj_b(a) = ((a·b) / ||b||^2) * b
        
        # 计算点积 (a·b)，形状为 (B, 1)
        dot_product = torch.sum(f_visual * p_text, dim=-1, keepdim=True)
        
        # 计算 ||b||^2，形状为 (B, 1)
        p_text_norm_sq = torch.sum(p_text * p_text, dim=-1, keepdim=True)
        
        # 计算投影系数，形状为 (B, 1)
        projection_scalar = dot_product / (p_text_norm_sq + 1e-8) # 加 epsilon 防止除以零
        
        # 乘以 p_text 得到投影向量 F_aligned
        f_aligned = projection_scalar * p_text # 形状: (B, D)
        
        # --- 步骤 2: 残余提取 (Residual Extraction) ---
        # 从原始视觉特征中减去对齐部分，得到正交的残余部分。
        f_residual = f_visual - f_aligned # 形状: (B, D)
        
        # --- 步骤 3: 残余门控 (Residual Gating) ---
        # 将 F_aligned 和 F_residual 拼接作为门控网络的输入。
        gate_input = torch.cat([f_aligned, f_residual], dim=-1) # 形状: (B, 2*D)
        
        # 通过门控网络计算逐维度的门控权重 g。
        g = self.gate_mlp(gate_input) # 形状: (B, D)
        
        # --- 步骤 4: 特征重组 (Feature Recomposition) ---
        # 使用门控权重 g 过滤残余特征。
        filtered_residual = g * f_residual # 形状: (B, D)
        
        # 将对齐特征和过滤后的残余特征相加。
        recomposed_feature = f_aligned + filtered_residual # 形状: (B, D)
        
        # 对最终的特征进行层归一化。
        f_semantic = self.layer_norm(recomposed_feature) # 形状: (B, D)
        
        return f_semantic

