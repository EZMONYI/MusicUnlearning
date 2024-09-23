from model.songmass import build_songmass
import time
import copy
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch
import sys
import numpy as np
from argparse import Namespace
import torch.nn.functional as F

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    new_lr = opt.sgda_learning_rate
    if steps > 0:
        new_lr = opt.sgda_learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    return new_lr


def train_distill(epoch, train_loader, module_list, swa_model, criterion_list, optimizer, opt, split, quiet=False):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()


    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    kd_losses = AverageMeter()
    top1 = AverageMeter()


    end = time.time()
    for idx, sample in enumerate(train_loader):
        print(idx)
        avged_loss = torch.zeros(1)
        count = 0
        for lang_pair in sample.keys():
            sample_key = lang_pair
            if sample_key.startswith('mass'):
                lang_pair = sample_key.split(':')[-1]
            net_input = sample[sample_key]['net_input']['src_tokens']
            net_output_s, logit_s, target_s = compute_forward(model_s, sample, sample_key, lang_pair)
            net_output_t, logit_t, target_t = compute_forward(model_t, sample, sample_key, lang_pair)
            # data_time.update(time.time() - end)


            # cls + kl div
            loss_cls = criterion_cls(logit_s, target_s)
            loss_div = criterion_div(logit_s, logit_t)

            # other kd beyond KL divergence
            if opt.distill == 'kd':
                loss_kd = 0
            else:
                raise NotImplementedError(opt.distill)

            if split == "minimize":
                loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
            elif split == "maximize":
                loss = -loss_div

            avged_loss += loss
            count += 1
        avged_loss /= count 
        if split == "minimize" and not quiet:
            losses.update(loss.item(), net_input.size(0))
        elif split == "maximize" and not quiet:
            kd_losses.update(loss.item(), net_input.size(0))


        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        if not quiet:
            if split == "mainimize":
                if idx % opt.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                        epoch, idx, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses))
                    sys.stdout.flush()

    
    if split == "minimize":
        if not quiet:
            print(' * Acc@1 {top1.avg:.3f} '
                  .format(top1=top1))

        return top1.avg, losses.avg
    else:
        return kd_losses.avg


def compute_forward(model, sample, sample_key):
    net_output = model(**sample[sample_key]['net_input'])
    logits = model.get_normalized_probs(net_output, log_probs=True).transpose(1, 2) 
    return net_output, logits


def unlearn_songmass():
    model, unlearn_set, retain_set = build_songmass('scrub_epoch_1.pt')
    unlearn_loader = DataLoader(unlearn_set, shuffle=True, batch_size=8, collate_fn=unlearn_set.collater)
    retain_loader = DataLoader(retain_set, shuffle=True, batch_size=8, collate_fn=retain_set.collater)
    teacher = copy.deepcopy(model)
    student = copy.deepcopy(model)
    module_list = nn.ModuleList([])
    module_list.append(teacher)
    module_list.append(student)

    trainable_list = nn.ModuleList([])
    trainable_list.append(student)

    scrub_args = Namespace(print_freq = 20, optim = 'sgd', gamma = 0.99,alpha = 0.001, beta = 0, smoothing = 0.0, msteps = 2, clip = 0.2, sstart = 10, kd_T = 4, distill = 'kd', sgda_batch_size = 128, del_batch_size = 32, sgda_epochs = 3, sgda_learning_rate = 0.0005, lr_decay_epochs = [3,5,9], lr_decay_rate = 0.1, sgda_weight_decay = 5e-4, sgda_momentum = 0.9)

    optimizer = torch.optim.SGD(trainable_list.parameters(),
                            lr=scrub_args.sgda_learning_rate,
                            momentum=scrub_args.sgda_momentum,
                            weight_decay=scrub_args.sgda_weight_decay)



    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(scrub_args.kd_T)
    criterion_kd = DistillKL(scrub_args.kd_T)
    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    
    criterion_list.append(criterion_div)    
    criterion_list.append(criterion_kd)     

    acc_rs = []
    acc_fs = []
    acc_vs = []
    acc_ts = []

    for epoch in range(1, scrub_args.sgda_epochs + 1):
            # train for one epoch
        lr = adjust_learning_rate(epoch, scrub_args, optimizer)
        print("==> SCRUB unlearning ...")

        maximize_loss = 0
        if epoch <= scrub_args.msteps:
            maximize_loss = train_distill(epoch, unlearn_loader, module_list, None, criterion_list, optimizer, scrub_args, "maximize")
        train_acc, train_loss = train_distill(epoch, retain_loader, module_list, None, criterion_list, optimizer, scrub_args, "minimize")

        
        print("maximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {}".format(maximize_loss, train_loss, train_acc))
                


if __name__ == "__main__":
    unlearn_songmass()