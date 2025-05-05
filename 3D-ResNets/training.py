import torch
import time
import os
import sys

import torch
import torch.distributed as dist

from utils import AverageMeter, calculate_accuracy


def train_epoch(epoch,
                data_loader,
                model,
                criterion,
                optimizer,
                device,
                current_lr,
                epoch_logger,
                batch_logger,
                tb_writer=None,
                distributed=False):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        # inputs *= 10
        # targets2 = (torch.round(targets).long()).to(device, non_blocking=True)
        optimizer  = torch.optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=1e-4, betas=(0.9, 0.999))
        targets= [float(x) for x in targets]
        targets = torch.tensor(targets)
        targets1 = (targets).to(device, non_blocking=True)
        outputs1 = model(inputs).flatten()
        loss = criterion(outputs1, targets1)
        alpha = 0
        # loss = loss1 + alpha * loss2
        # predicted_label = torch.argmax(outputs2, dim = 1)
        # outputs2 = (outputs2).to(device, non_blocking=True)
        # acc = calculate_accuracy(outputs2, targets2)
        # print("train_output2 is ",predicted_label,'\n',"train_target2 is ",targets2,"\n","train_loss2 is ",loss2,"\n","acc is ",acc,"\n")
        print("train_output is ",outputs1,'\n',"train_target is ",targets1,"\n","train_loss is ",loss,"\n")
        # acc = calculate_accuracy(outputs, targets)

        losses.update(loss.item(), inputs.size(0))
        # accuracies.update(acc, inputs.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # if batch_logger is not None:
        #     batch_logger.log({
        #         'epoch': epoch,
        #         'batch': i + 1,
        #         'iter': (epoch - 1) * len(data_loader) + (i + 1),
        #         'loss': losses.val,
        #         # 'acc': accuracies.val,
        #         'lr': current_lr
        #     })

        # print('Epoch: [{0}][{1}/{2}]\t'
        #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #       'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(epoch,
        #                                                  i + 1,
        #                                                  len(data_loader),
        #                                                  batch_time=batch_time,
        #                                                  data_time=data_time,
        #                                                  loss=losses,
        #                                                  acc=accuracies))
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch,
                                                         i + 1,
                                                         len(data_loader),
                                                         batch_time=batch_time,
                                                         data_time=data_time,
                                                         loss=losses))

    if distributed:
        loss_sum = torch.tensor([losses.sum],
                                dtype=torch.float32,
                                device=device)
        loss_count = torch.tensor([losses.count],
                                  dtype=torch.float32,
                                  device=device)
        # acc_sum = torch.tensor([accuracies.sum],
        #                        dtype=torch.float32,
        #                        device=device)
        # acc_count = torch.tensor([accuracies.count],
        #                          dtype=torch.float32,
        #                          device=device)

        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
        # dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
        # dist.all_reduce(acc_count, op=dist.ReduceOp.SUM)

        losses.avg = loss_sum.item() / loss_count.item()
        # accuracies.avg = acc_sum.item() / acc_count.item()

    # if epoch_logger is not None:
    #     epoch_logger.log({
    #         'epoch': epoch,
    #         'loss': losses.avg,
    #         # 'acc': accuracies.avg,
    #         'lr': current_lr
    #     })
    
    print('epoch:{0}\t'
        'loss:{1}\t'
        'lr{2}\t'.format(epoch,losses.avg,current_lr))
    # if tb_writer is not None:
    #     tb_writer.add_scalar('train/loss', losses.avg, epoch)
        # tb_writer.add_scalar('train/acc', accuracies.avg, epoch)
        # tb_writer.add_scalar('train/lr', accuracies.avg, epoch)
