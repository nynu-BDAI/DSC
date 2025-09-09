# import new Network name here and add in model_class args
#from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F

from losses import SupContrastive


def base_train(model, trainloader, criterion, optimizer, scheduler, epoch,args):

    #视觉扰动负样本
    text_feats_np = np.load(f"/mnt/Data/wangshilong/SAVC/text_generated/{args.dataset}_all_classText_feature.npy")
    class_text_feats = torch.from_numpy(text_feats_np).float().to("cuda") 
    class_text_feats = F.normalize(class_text_feats, dim=-1)
    class_text_feats.requires_grad_(False)  #cifar100: 100*1*512
    #NUM_CLASSES, FEAT_DIM = class_text_feats.shape

    tl = Averager()
    tl_joint = Averager()
    tl_moco = Averager()
    tl_moco_global = Averager()
    tl_moco_small = Averager()
    ta = Averager()
    model = model.train()
    tqdm_gen = tqdm(trainloader)
    #一个epoch中所有batch进行计算
    for i, batch in enumerate(tqdm_gen, 1):
        if args.use_text:
            data, single_labels,text = [_ for _ in batch]
            b, c, h, w = data[1].shape
            original = data[0].cuda(non_blocking=True)
            data[1] = data[1].cuda(non_blocking=True) #两个全局视图 b*3*224*224
            data[2] = data[2].cuda(non_blocking=True)
            single_labels = single_labels.cuda(non_blocking=True) # b
            if len(args.num_crops) > 1:
                data_small = data[args.num_crops[0]+1].unsqueeze(1) #batch*1*c*h*w
                for j in range(1, args.num_crops[1]):
                    data_small = torch.cat((data_small, data[j+args.num_crops[0]+1].unsqueeze(1)), dim=1) #b*4*c*h*w
                data_small = data_small.view(-1, c, 224, 224).cuda(non_blocking=True)
            else:
                data_small = None
            
            data_classify = original #b*c*h*w
            data_query = data[1]
            data_key = data[2]
            data_small = data_small #4b*c*h*w
            text=text
            
            preds, output_global, output_small, target_global, target_small,cos_sim, push_loss = model(im_cla=data_classify, im_q=data_query, im_k=data_key, labels=single_labels, im_q_small=data_small,text=text,class_text_feats=class_text_feats)

            """ preds:原始图像与分类器权重的余弦相似度得分 b*100
                output_global:大crop与k 以及队列中所有图像的余弦相似度得分 b*8193
                output_small:小crop与k 以及队列中所有图像的余弦相似度得分 4b*8193
                targets_global:大crop的标签 b*8193
                targets_small:小crop的标签 4b*8193
            """
            #对比损失
            loss_moco_global = criterion(output_global, target_global)
            loss_moco_small = criterion(output_small, target_small)
            loss_moco = args.alpha * loss_moco_global + args.beta * loss_moco_small

            #分类损失
            preds = preds[:, :args.base_class] #b*num_base_class 这个样本与当前所有类别的相似度得分
            joint_loss = F.cross_entropy(preds, single_labels) # 最大化每个样本在其标签位置的预测值。

            #对齐损失
            loss_t2i= 1-cos_sim.mean()
            loss = joint_loss + loss_moco + loss_t2i+ push_loss
            total_loss = loss
            
            acc = count_acc(preds, single_labels, epoch)
            lrc = scheduler.get_last_lr()[0] #当前学习率

            #进度条的前缀表示
            tqdm_gen.set_description('Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
            #一个batch的数据
            tl.add(total_loss.item())
            tl_joint.add(joint_loss.item())
            tl_moco_global.add(loss_moco_global.item())
            tl_moco_small.add(loss_moco_small.item())
            tl_moco.add(loss_moco.item())
            ta.add(acc)

            #优化三步曲
            optimizer.zero_grad()#清空上次计算的旧梯度
            loss.backward()#反向传播
            optimizer.step()#根据梯度更新参数
        else:
            data, single_labels = [_ for _ in batch]
            b, c, h, w = data[1].shape
            original = data[0].cuda(non_blocking=True)
            data[1] = data[1].cuda(non_blocking=True) #两个全局视图 b*3*224*224
            data[2] = data[2].cuda(non_blocking=True)
            single_labels = single_labels.cuda(non_blocking=True) # b
            if len(args.num_crops) > 1:
                data_small = data[args.num_crops[0]+1].unsqueeze(1) #batch*1*c*h*w
                for j in range(1, args.num_crops[1]):
                    data_small = torch.cat((data_small, data[j+args.num_crops[0]+1].unsqueeze(1)), dim=1) #b*4*c*h*w
                data_small = data_small.view(-1, c, 224, 224).cuda(non_blocking=True)
            else:
                data_small = None
            
            data_classify = original #b*c*h*w
            data_query = data[1]
            data_key = data[2]
            data_small = data_small #4b*c*h*w
            
            preds, output_global, output_small, target_global, target_small = model(im_cla=data_classify, im_q=data_query, im_k=data_key, labels=single_labels, im_q_small=data_small)

            #对比损失
            loss_moco_global = criterion(output_global, target_global)
            loss_moco_small = criterion(output_small, target_small)
            loss_moco = args.alpha * loss_moco_global + args.beta * loss_moco_small

            #分类损失
            preds = preds[:, :args.base_class] #b*num_base_class 这个样本与当前所有类别的相似度得分
            joint_loss = F.cross_entropy(preds, single_labels) # 最大化每个样本在其标签位置的预测值。

            loss = joint_loss + loss_moco
            total_loss = loss
            
            acc = count_acc(preds, single_labels, epoch)
            lrc = scheduler.get_last_lr()[0] #当前学习率

            #进度条的前缀表示
            tqdm_gen.set_description('Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
            #一个batch的数据
            tl.add(total_loss.item())
            tl_joint.add(joint_loss.item())
            tl_moco_global.add(loss_moco_global.item())
            tl_moco_small.add(loss_moco_small.item())
            tl_moco.add(loss_moco.item())
            ta.add(acc)

            #优化三步曲
            optimizer.zero_grad()#清空上次计算的旧梯度
            loss.backward()#反向传播
            optimizer.step()

    #一个epoch数据
    tl = tl.item()
    ta = ta.item()
    tl_joint = tl_joint.item()
    tl_moco = tl_moco.item()
    tl_moco_global = tl_moco_global.item()
    tl_moco_small = tl_moco_small.item()
    return tl, tl_joint, tl_moco, tl_moco_global, tl_moco_small, ta


def replace_base_fc(trainset, test_transform,model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = test_transform #希望得到标准状态下的特征而非训练中各种增强
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            if len(batch)==2:
                data, label = [_ for _ in batch]
            elif len(batch)==3:
                data, label, text = [_ for _ in batch]
            b = data.size()[0]
            data = data.cuda()
            labels=label.cuda()
            model.mode = 'encoder'
            embedding = model(data)#直接获取特征

            embedding_list.append(embedding.cpu()) #将每个batch图像特征放到embedding_list里。持续的积累大量张量在cpu上更加安全
            label_list.append(labels.cpu())
    embedding_list = torch.cat(embedding_list, dim=0) #tensor类型
    label_list = torch.cat(label_list, dim=0) #tensor类型

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero() #当 class_index=0时，从 label_list 中找出所有标签为0的样本索引, 这个索引在 embedding list 中就是对应的标签为 0 的样本特征
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0) #均值
        proto_list.append(embedding_this) #0-119

    proto_list = torch.stack(proto_list, dim=0) #num_class * dim

    model.fc.weight.data[:args.base_class] = proto_list #将原始模型的fc中base session对应的weight更换为原型表示

    return model


def update_fc_ft(trainloader,  model,  session, args):
    # incremental finetuning
    old_class = args.base_class + args.way * (session - 1)
    new_class = args.base_class + args.way * session
    "上述两句代码的作用：计算出当前增量会话中，新类别在整个分类头权重矩阵中的起始和结束索引；为了精确地定位到需要操作的权重部分"
    new_fc = nn.Parameter(
        torch.rand(args.way, model.num_features, device="cuda"),
        requires_grad=True)
    new_fc.data.copy_(model.fc.weight[old_class : new_class, :].data) #而这部分权重，是在上一个函数 update_fc 中，通过计算新类别数据的“平均特征向量”（即原型）刚刚被初始化的。
    """
        因为用随机权重作为微调的起点是一个很差的选择。我们已经通过 update_fc 函数计算出了一个非常好的初始值（类别原型）
        所以，这行代码的作用是用这个更好的“原型”值，来初始化我们即将要微调的 new_fc 参数。这能让微调过程起点更高、收敛更快、结果更稳定。
    """
    if args.dataset == 'mini_imagenet':
        optimizer = torch.optim.SGD([{'params': new_fc, 'lr': args.lr_new},
                                     {'params': model.lepc_network.gate_mlp[0].weight, 'lr':args.lr_new},
                                     {'params': model.lepc_network.gate_mlp[0].bias, 'lr':args.lr_new},
                                     {'params': model.lepc_network.gate_mlp[2].weight, 'lr':args.lr_new},
                                     {'params': model.lepc_network.layer_norm.weight, 'lr':args.lr_new},
                                     {'params': model.lepc_network.layer_norm.bias, 'lr':args.lr_new}],
                                    momentum=0.9, dampening=0.9, weight_decay=0)
        
    if args.dataset == 'cub200':
        optimizer = torch.optim.SGD([{'params': new_fc, 'lr': args.lr_new},
                                     {'params': model.lepc_network.gate_mlp[0].weight, 'lr':args.lr_new},
                                     {'params': model.lepc_network.gate_mlp[0].bias, 'lr':args.lr_new},
                                     {'params': model.lepc_network.gate_mlp[2].weight, 'lr':args.lr_new},
                                     {'params': model.lepc_network.layer_norm.weight, 'lr':args.lr_new},
                                     {'params': model.lepc_network.layer_norm.bias, 'lr':args.lr_new}],
                                    momentum=0.9, dampening=0.9, weight_decay=0)
        
    elif args.dataset == 'cifar100':
        optimizer = torch.optim.Adam([{'params': new_fc, 'lr': args.lr_new},
                                     {'params': model.lepc_network.gate_mlp[0].weight, 'lr':args.lr_new},
                                     {'params': model.lepc_network.gate_mlp[0].bias, 'lr':args.lr_new},
                                     {'params': model.lepc_network.gate_mlp[2].weight, 'lr':args.lr_new},
                                     {'params': model.lepc_network.layer_norm.weight, 'lr':args.lr_new},
                                     {'params': model.lepc_network.layer_norm.bias, 'lr':args.lr_new}],
                                      weight_decay=0)
    
    criterion = SupContrastive().cuda() 

    with torch.enable_grad():
        for epoch in range(args.epochs_new):
            tqdm_gen = tqdm(trainloader)
            for _,batch in enumerate(tqdm_gen,1):
                data, single_labels,text = [_ for _ in batch]
                b, c, h, w = data[1].shape
                origin = data[0].cuda(non_blocking=True)
                data[1] = data[1].cuda(non_blocking=True)
                data[2] = data[2].cuda(non_blocking=True)
                single_labels = single_labels.cuda(non_blocking=True)
                if len(args.num_crops) > 1:
                    data_small = data[args.num_crops[0]+1].unsqueeze(1)
                    for j in range(1, args.num_crops[1]):
                        data_small = torch.cat((data_small, data[j+args.num_crops[0]+1].unsqueeze(1)), dim=1)
                    data_small = data_small.view(-1, c, 224, 224).cuda(non_blocking=True)
                else:
                    data_small = None
            data_classify = origin 
            data_query =data[1]
            data_key = data[2]
            data_small = data_small
            joint_labels =single_labels
            
            old_fc = model.fc.weight[:old_class, :].clone().detach()
            '''
            做什么：创建一份对应所有旧类别的分类器权重的副本，并使用 .detach() 将其从计算图中分离。
            为什么这样做：这是为了**“冻结”旧知识**。在计算分类损失时，我们需要一个包含所有新旧类别的完整分类器。
            但是，我们绝不希望在这次微调中更新旧类别的权重（这会导致灾难性遗忘）。
            .detach() 操作告诉PyTorch：“不要追踪这个 old_fc 张量的梯度”，从而确保在反向传播时，梯度不会流向这部分权重。
            '''
            fc = torch.cat([old_fc, new_fc], dim=0) #将冻结的旧权重和正在训练的新权重拼接

            #分类损失
            # features= model.encode_q(data_classify)
            # features.detach() #joint_loss产生的梯度无法传回encode_q,只用于更新new_fc
            # logits = model.get_logits(features,fc)#计算分类得分
            # joint_loss = F.cross_entropy(logits, joint_labels)

            logits = model.forward_metric_LEPC(data_classify,text,fc)
            joint_loss = F.cross_entropy(logits, joint_labels)

            _, output_global, output_small, target_global, target_small,cos_sim = model(im_cla=data_classify, im_q=data_query, im_k=data_key, labels=joint_labels, im_q_small=data_small, base_sess=False, last_epochs_new=(epoch==args.epochs_new-1),text=text, class_text_feats=None)
            loss_moco_global = criterion(output_global, target_global)
            loss_moco_small = criterion(output_small, target_small)
            loss_moco = args.alpha * loss_moco_global + args.beta * loss_moco_small 
            loss_t2i=1-cos_sim.mean()
            loss = joint_loss + loss_moco + loss_t2i
            
            # for i, param_group in enumerate(optimizer.param_groups):
            #     print(f"Param group {i}:")
            #     for param in param_group['params']:
            #         print(param.shape, param.requires_grad)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    #训完之后把训好的针对新类别的权重更新回主模型
    "old_class 是新加入的虚拟类别的起始索引,new_class是它们的结束索引,这个切片操作精确地选中了权重矩阵中只属于当前增量会话新类别的那几行。旧类别和其他未来类别的权重行则完全不受影响"
    model.fc.weight.data[old_class : new_class, :].copy_(new_fc.data)

def test(model, testloader, epoch, args, session):
    test_class = args.base_class + session * args.way #动态的计算测试阶段参与测试的类别数目
    model = model.eval()
    vl = Averager() #loss
    va = Averager() #acc
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            b = data.size()[0]
            data = data #原始图像，不含crop
            all_preds = model(data) #data进入 encoder 然后让特征和分类器权重进行点积得到分类得分
            preds = all_preds[:, :test_class] #我知道你（模型）对所有可能的类别都做出了预测，但我现在只关心你对那些我们已经学过的类别的预测表现如何。这也是为什么需要前面的 test_class 计算
        
            loss = F.cross_entropy(preds, test_label)
            acc = count_acc(preds, test_label, epoch)

            vl.add(loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    return vl,va