from .base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy
from .Network_noFan import MYNET
from .helper_noFan import *
from utils import *
from dataloader.data_utils import *
from losses import SupContrastive
from augmentations import fantasy


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args) #调用父类 Trainer 的构造函数 __init__
        self.args = args
        self.set_save_path() #更新了保存路径
        self.args = set_up_datasets(self.args)

        "不使用虚拟变化"
        # if args.fantasy is not None:
        #     # x=fantasy.__dict__
        #     self.transform, self.num_trans = fantasy.__dict__[args.fantasy]() #__dict__ 是Python中的内置属性，它返回对象的属性字典。
        # else:
        #     self.transform = None
        #     self.num_trans = 0

        self.model = MYNET(self.args, mode=self.args.base_mode)
        #self.model = nn.DataParallel(self.model, list(map(int, args.gpu.split(','))))
        self.model = self.model.cuda()

        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

        

    def get_optimizer_base(self):

        optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.decay)
        if self.args.schedule == 'Step': #步长式学习率衰减，每经过多少epoch lr*gamma
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone': #多步长式学习率衰减，milestones【60.80】，当到epoch为60和80时乘以gamma
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)

        return optimizer, scheduler

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader
        
    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        for session in range(args.start_session, args.sessions): #start_session:0 cifar100_session:9

            train_set, trainloader, testloader = self.get_dataloader(session) #不同 session 选择不同的数据

            self.model.load_state_dict(self.best_model_dict) #确保增量阶段用到的是最好的模型

            if session == 0:  # load base class train img label
                
                train_set.multi_train = True #训练的时候 dataset返回的 image 包括原图还有裁剪的小图
                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()
                criterion = SupContrastive() 
                criterion = criterion.cuda()
                
                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    # train base sess
                    tl, tl_joint, tl_moco, tl_moco_global, tl_moco_small, ta = base_train(self.model, trainloader, criterion, optimizer, scheduler, epoch, args)
                    # test model with all seen class 训一轮测一轮
                    tsl, tsa = test(self.model, testloader, epoch, args, session) #testloader中 multi_train=false, 传入 session 是为了获取本次参加测试的类别数目

                    # save better model
                    if (tsa * 100) >= self.trlog['max_acc'][session]: #basesession训练，如果当前轮次的测试结果比原来的好
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100)) #在session 0的位置写入
                        self.trlog['max_acc_epoch'] = epoch #记录结果最好的轮次 int类型
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth') #args.save_path来自于初始化中的self.set_sava_path
                        torch.save(dict(params=self.model.state_dict()), save_model_dir) #保存模型到磁盘
                        torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))#保存优化器状态，以便再恢复训练的时候可以快速适应
                        self.best_model_dict = deepcopy(self.model.state_dict())  #更新保存的最佳模型，到内存ram
                        print('********A better model is found!!**********')
                        print('Saving model to :%s' % save_model_dir)
                    print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                       self.trlog['max_acc'][session]))

                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)
                    self.trlog['test_loss'].append(tsl)
                    self.trlog['test_acc'].append(tsa)
                    lrc = scheduler.get_last_lr()[0] #当前有效的学习率
                    result_list.append(
                    'epoch:%03d,lr:%.4f,training_loss:%.5f,joint_loss:%.5f, moco_loss:%.5f, moco_loss_global:%.5f, moco_loss_small:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f'% (epoch, lrc, tl, tl_joint, tl_moco, tl_moco_global, tl_moco_small, ta, tsl, tsa))    
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish this session' % (
                                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                    scheduler.step() #当前epoch结束更新学习率，具体要看用的是什么scheduler，有的是在处理每个batch后进行处理

                # epoch跑完了
                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))

           
                #上面的 test 结果是图像与学习到的分类器权重得到的，下面是通过训练集的原型替换分类器权重进行 test
                if not args.not_data_init: 
                    self.model.load_state_dict(self.best_model_dict)
                    train_set.multi_train = False
                    self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)#这一步就将原始模型的fc的权重换为原型表示
                    best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                    print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    torch.save(dict(params=self.model.state_dict()), best_model_dir) #覆盖之前fc训练得到的权重

                    self.model.mode = 'avg_cos' #原型模式
                    tsl, tsa = test(self.model, testloader, 0, args, session)
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))
                        

            else:  # incremental learning sessions
                print("training session: [%d]" % session)
                self.model.mode = self.args.new_mode #默认为 avg_cos
                self.model.eval()
                train_transform = trainloader.dataset.transform #cifar.py中的 if self.train : self.transform
                trainloader.dataset.transform = testloader.dataset.transform # else : self.transform
                self.model.update_fc(trainloader, np.unique(train_set.targets), session) #增量阶段的训练集去更新新类别的分类头

                if args.incft:
                    trainloader.dataset.transform = train_transform
                    train_set.multi_train = True #使用crop
                    update_fc_ft(trainloader, self.model,session, args) 

                tsl, tsa = test(self.model, testloader, 0, args, session)#用截止目前所有的类别样本来测试

                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                torch.save(dict(params=self.model.state_dict()), save_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))
        #所有阶段跑完
        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)
        
    def set_save_path(self):
        mode = self.args.base_mode + '-' + self.args.new_mode #ft_cos-avg_cos
        if not self.args.not_data_init: #false
            mode = mode + '-' + 'data_init'

        self.args.save_path = '%s/' % self.args.dataset # '%s/' 是一个模板字符串。其中 %s 是一个占位符，表示这里将被一个字符串（string）替换。后面的 / 是一个普通的斜杠字符，用来创建文件夹层级.%:是格式化操作符。它会把右边的变量（self.args.dataset）的值，填充到左边字符串的 %s 占位符中
        self.args.save_path = self.args.save_path + '%s/' % self.args.project

        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session) #%d 十进制整数占位符
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, mile_stone, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Cosine':
            self.args.save_path = self.args.save_path + 'Cosine-Epo_%d-Lr_%.4f' % (
                self.args.epochs_base, self.args.lr_base)
            
        if 'cos' in mode:
            self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.temperature)

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)
        self.args.save_path = self.args.save_path + f'-fantasy_{self.args.fantasy}'
        self.args.save_path = self.args.save_path + '-alpha_%.2f-beta_%.2f' % (self.args.alpha, self.args.beta)
        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path) #确认一下是否存在，不存在的话创建
        return None
