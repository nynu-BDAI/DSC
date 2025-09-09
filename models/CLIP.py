
import torch
import torch.nn as nn
import clip
import torch.nn.functional as F

class CLIPEncoder(nn.Module):
    def __init__(self, device, dim):
        super(CLIPEncoder, self).__init__()
        self.device = device
        self.model, _ = clip.load("ViT-B/32", device=device)
        self.model.float()
        self.encoder = self.model.visual

        #文本分支训练策略
        for param in self.model.transformer.parameters():
            param.requires_grad = False
        for param in self.model.token_embedding.parameters():
            param.requires_grad = False
        self.model.ln_final.weight.requires_grad = False
        self.model.ln_final.bias.requires_grad = False  
        self.model.text_projection.requires_grad = False

        #视觉分支训练策略
        for param in self.encoder.parameters():
            param.requires_grad = False
        for name, param in self.encoder.named_parameters():
            if "ln_" in name or "transformer.resblocks.10" in name or "transformer.resblocks.11" in name:
                param.requires_grad = True


        self.out_dim = dim 

    def forward(self, x):
        if self.out_dim==512:
            y = self.encoder(x)
        return y  
    def forward_text(self, x):
        x= clip.tokenize(x,truncate=True).to(self.device)
        y = self.model.encode_text(x)
        return y
        
def CLIP (**kwargs):
    model = CLIPEncoder(device='cuda',**kwargs)
    return model