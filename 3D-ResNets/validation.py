import torch
import time
import sys

import torch
import torch.distributed as dist

from utils import AverageMeter, calculate_accuracy


def val_epoch(epoch,
              data_loader,
              model,
              criterion,
              device,
              logger,
              tb_writer=None,
              distributed=False):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # accuracies = AverageMeter()

    end_time = time.time()

    with torch.no_grad():
        m_val_loss = 0
        num = 0
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)
            # inputs *= 10
            # targets = torch.Tensor(targets)
            # targets = (targets*10).to(device, non_blocking=True)
            # outputs = model(inputs)
            # loss = criterion(outputs/10, targets/10)
            # print("val_output is ",outputs/10,'\n',"val_target is ",targets/10,"\n","val_loss is ",loss,"\n")
            # targets2 = (torch.round(targets).long()).to(device, non_blocking=True)
            targets= [float(x) for x in targets]
            targets = torch.tensor(targets)
            targets1 = (targets).to(device, non_blocking=True)
            outputs1 = model(inputs)
            loss = criterion(outputs1, targets1)
            m_val_loss += loss.item()
            num += 1
            print("val_output is ",outputs1,'\n',"val_target is ",targets1,"\n","val_loss is ",loss,"\n")
            # loss2 = criterion2(outputs2, targets2)
            # alpha = 0
            # loss = alpha * loss1 + loss2
            # predicted_label = torch.argmax(outputs2, dim = 1)
            # outputs2 = (outputs2).to(device, non_blocking=True)
            # acc = calculate_accuracy(outputs2, targets2)
            # print("val_output2 is ",predicted_label,'\n',"val_target2 is ",targets2,"\n","val_loss2 is ",loss2,"\n","acc is ",acc,"\n")
            # acc = calculate_accuracy(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            # accuracies.update(acc, inputs.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            # print('Epoch: [{0}][{1}/{2}]\t'
            #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
            #           epoch,
            #           i + 1,
            #           len(data_loader),
            #           batch_time=batch_time,
            #           data_time=data_time,
            #           loss=losses))
                    #   acc=accuracies))
        m_val_loss /= num
        print("n_val_loss is ",m_val_loss,'\n')

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

    # if logger is not None:
    #     logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

    if tb_writer is not None:
        tb_writer.add_scalar('val/loss', losses.avg, epoch)
        # tb_writer.add_scalar('val/acc', accuracies.avg, epoch)

    return losses.avg
