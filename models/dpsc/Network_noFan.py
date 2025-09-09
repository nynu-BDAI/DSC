import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from models.CLIP import *
from .LEPC_Network import *  

class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()
        self.lepc_network=LEPC_NET(feature_dim=512, hidden_dim_ratio=0.5)
        self.mode = mode
        self.args = args
        if self.args.dataset in ['cifar100']:
            self.encoder_q= CLIP(dim=512) 
            # for name,_ in self.encoder_q.named_parameters():
            #     print(name)
            self.encoder_k= CLIP(dim=512)
            self.num_features = 512
        if self.args.dataset in ['mini_imagenet']:
            self.encoder_q = CLIP(dim=512)  
            self.encoder_k = CLIP(dim=512) 
            self.num_features = 512
        if self.args.dataset == 'cub200':
            self.encoder_q = CLIP(dim=512) 
            self.encoder_k = CLIP(dim=512) 
            self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #全局平均池化（b*64*8*8）变为（b*64*1*1）
        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)
            
        self.K = self.args.moco_k #队列长度  65536
        self.m = self.args.moco_m #动量更新系数，用于缓慢更新encoder_k 0.995
        self.T = self.args.moco_t #mcoco对比学习的温度系数 0.07
        
        if self.args.mlp:  # hack: brute-force replacement
            self.encoder_q.fc = nn.Sequential(nn.Linear(self.num_features, self.num_features), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(self.num_features, self.num_features), nn.ReLU(), self.encoder_k.fc)
#           如果为true，则将编码器中的最后一层fc进行增强，原始的一层变为两层

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
        # create the queue
        self.register_buffer("queue", torch.randn(512, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))#队列指针
        self.register_buffer("label_queue", torch.zeros(self.K).long() - 1)

                    
    @torch.no_grad() #这里更新的是encoder_k,是随encoder_q更新的，因此应该是手动的根据动量更新，而encoder_q是自动更新的需要梯度传播
    def _momentum_update_key_encoder(self, base_sess):
        """
        Momentum update of the key encoder
        """
        if base_sess:
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        else:
            for k, v in self.encoder_q.named_parameters():
                if "transformer.resblocks.10" in k or "transformer.resblocks.11" in k:
                    self.encoder_k.state_dict()[k].data = self.encoder_k.state_dict()[k].data * self.m + v.data * (1. - self.m)
                    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
 
        # replace the keys and labels at ptr (dequeue and enqueue)
        if ptr + batch_size > self.K:
            remains = ptr + batch_size - self.K
            self.queue[:, ptr:] = keys.T[:, :batch_size - remains]
            self.queue[:, :remains] = keys.T[:, batch_size - remains:]
            self.label_queue[ptr:] = labels[ :batch_size - remains]
            self.label_queue[ :remains] = labels[batch_size - remains:]
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T # this queue is feature queue
            self.label_queue[ptr:ptr + batch_size] = labels        
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr
    
    def forward_metric(self, x): #im_cla
        x = self.encode_q(x)
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1)) #归一化的x与归一化的分类器的余弦相似度，p=2表示L2范数
            x = self.args.temperature  * x

        elif 'dot' in self.mode:
            x = self.fc(x)

        return x # joint, contrastive

    def encode_q(self, x):
        y = self.encoder_q(x)
       
        return y
    
    def encode_k(self, x):
        y = self.encoder_k(x)
        
        return y

    def forward(self, im_cla, im_q=None, im_k=None, labels=None, im_q_small=None, base_sess=True, 
                last_epochs_new=False, text=None, class_text_feats=None):
        
        if self.mode != 'encoder': 
            if im_q == None: #test
                x = self.forward_metric(im_cla) #
                return x
            else:
                b = im_q.shape[0] 
                logits_classify = self.forward_metric(im_cla) #原始图像与分类器的权重余弦相似度得分 b*100, 100:100个类别方向    
                q = self.encode_q(im_q) #b*512

                q = nn.functional.normalize(q, dim=1)
                feat_dim = q.shape[-1] #512
                q = q.unsqueeze(1) #b*1*512

                if im_q_small is not None:
                    q_small = self.encode_q(im_q_small) #4b*512
                    q_small = q_small.view(b, -1, feat_dim)  # b*4*512
                    q_small = nn.functional.normalize(q_small, dim=-1)

                # compute key features
                with torch.no_grad():  # no gradient to keys
                    self._momentum_update_key_encoder(base_sess)  # update the key encoder
                    k = self.encode_k(im_k)  # keys: b*512
                    k = nn.functional.normalize(k, dim=1)

                # compute logits
                # Einstein sum is more intuitive
                # positive logits: Nx1
                q_global = q #b*1*512
                l_pos_global = (q_global * k.unsqueeze(1)).sum(2).view(-1, 1) #每个q和它对应k的内积 b*1
                l_pos_small = (q_small * k.unsqueeze(1)).sum(2).view(-1, 1) #每个小crop和它所属k的内积 4b*1

                # negative logits: NxK
                l_neg_global = torch.einsum('nc,ck->nk', [q_global.view(-1, feat_dim), self.queue.clone().detach()]) #einsum等价于矩阵乘法 q与所有队列中样本的相似度 size：b*8192
                l_neg_small = torch.einsum('nc,ck->nk', [q_small.view(-1, feat_dim), self.queue.clone().detach()]) #4b*8192 4个小图与队列中所有样本的相似度

                # logits: Nx(1+K)
                logits_global = torch.cat([l_pos_global, l_neg_global], dim=1) #b*8193 第一列是q 与 k，后续的是 q 与队列
                logits_small = torch.cat([l_pos_small, l_neg_small], dim=1) #4b*8193 同上不过是 q_small

                # apply temperature
                logits_global /= self.T
                logits_small /= self.T

                # one-hot target from augmented image
                positive_target = torch.ones((b, 1)).cuda() #创建 b*1的全1张量
                # find same label images from label queue
                # for the query with -1, all 
                targets = ((labels[:, None] == self.label_queue[None, :]) & (labels[:, None] != -1)).float().cuda() #b*8192
                targets_global = torch.cat([positive_target, targets], dim=1) #图像本身和自己是正样本，与队列中哪些还是正样本
                targets_small = targets_global.repeat_interleave(repeats=self.args.num_crops[1], dim=0) #将targets_global重复num_crops[1]次, 一个样本一个样本的复制
                
                # dequeue and enqueue
                if base_sess or (not base_sess and last_epochs_new): #如果是第一次学习一批基础类别时为True，或者是增量阶段且不是最后
                    self._dequeue_and_enqueue(k, labels)

                if text is None:   
                    return logits_classify, logits_global, logits_small, targets_global, targets_small   
            
                else:
                    #防遗忘第一步
                    with torch.no_grad():
                        text_feature = self.encoder_q.forward_text(text)
                    image_feature = self.encode_q(im_cla)
                    image_feature = image_feature / image_feature.norm(dim=1, keepdim=True)
                    text_feature = text_feature / text_feature.norm(dim=1, keepdim=True)
                    cos_sim=F.cosine_similarity(image_feature, text_feature, dim=-1)

                    #放遗忘第二步
                    current_text_feature = text_feature
                    current_image_feature = image_feature

                    if class_text_feats is not None:
                        with torch.no_grad():
                            class_text_feats = class_text_feats.squeeze(1)  # [100*1*512]-->[100, 512]
                            sim_all = current_text_feature @ class_text_feats.t() 
                            sim_all.scatter_(1, labels.view(-1,1), float("-inf"))    
                            topk = 5
                            _, idx = torch.topk(sim_all, k=topk, dim=1) 

                        neg_text_feature = class_text_feats[idx]   

                        current_image_feature=current_image_feature.unsqueeze(1)
                        current_text_feature=current_text_feature.unsqueeze(1)
                        
                        lam=1
                        dert=neg_text_feature - current_text_feature
                        img_neg = current_image_feature + lam * dert
                        img_neg = F.normalize(img_neg, dim=-1)        

                        pull_apart = F.cosine_similarity(current_image_feature, img_neg, dim=-1)  
                        loss_pull  = pull_apart.mean()  # 希望越小越好


                        return logits_classify, logits_global, logits_small, targets_global, targets_small, cos_sim,loss_pull
                    else:
                        
                        return logits_classify, logits_global, logits_small, targets_global, targets_small, cos_sim
            """
            logits_classify:原始图像与分类器权重的余弦相似度得分 b*100
            logits_global:大crop与k 以及队列中所有图像的余弦相似度得分 b*8193
            logits_small:小crop与k 以及队列中所有图像的余弦相似度得分 4b*8193
            targets_global:大crop的标签 b*8193
            targets_small:小crop的标签 4b*8193
            """       
        
        elif self.mode == 'encoder':
            x= self.encode_q(im_cla)
            return x
        else:
            raise ValueError('Unknown mode')
    
    
    def update_fc(self,dataloader,class_list,session):
        for batch in dataloader: 
            if len(batch)==2:
             data, label = [_ for _ in batch]
            elif len(batch)==3:
             data, label, text = [_ for _ in batch]
            b = data.size()[0]
            data = data.cuda() #(way*shot) * c * h * w
            labels =label.cuda() # (way*shot)
            data=self.encode_q(data)  #输入模型得到特征(way*shot)*512
            data.detach()#将提取的特征张量从计算图中分离，不要对data追踪梯度信息反向传播更新模型

        if self.args.not_data_init: #是否开启了not_data_init
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else: #✅
            new_fc = self.update_fc_avg(data, labels, class_list) #class_list=np.unique(targets)
            

    def update_fc_avg(self,data,labels,class_list): #更新增量类别分类头的权重为新类别计算的原型
        new_fc=[]
        for class_index in class_list: #对于每一个增量类
                index = class_index  #目标标签
                data_index=(labels==index).nonzero().squeeze(-1) #在数据中找到目标标签的索引
                embedding=data[data_index] #根据数据索引取对应数据
                proto=embedding.mean(0) #取均值
                new_fc.append(proto)
                self.fc.weight.data[index]=proto
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc 

    def get_logits(self,x,fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x,fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))
        
    def  forward_metric_LEPC(self,image,text,fc):
        lepc_net=self.lepc_network
        image_feature = self.encode_q(image)
        image_feature.detach() 
        text_feature = self.encoder_q.forward_text(text)
        enhance_image_feature=lepc_net(image_feature,text_feature)
        if 'cos' in self.mode:
            x = F.linear(F.normalize(enhance_image_feature, p=2, dim=-1), F.normalize(fc, p=2, dim=-1)) #归一化的x与归一化的分类器的余弦相似度，p=2表示L2范数
            x = self.args.temperature * x
        return x