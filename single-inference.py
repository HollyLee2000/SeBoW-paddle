import paddle
import paddle.vision.transforms as transforms
import argpar
import os
import copy
import paddle.nn as nn
from Tree_single_inference import ForestNet
from tensorboardX import SummaryWriter
import time
from utils.logger import Logger, savefig
from utils.misc import AverageMeter
from utils.util import accuracy, save_checkpoint, adjust_learning_rate2  # , accuracy_stage2
import paddle.vision.datasets as datasets
from dataset import CIFAR100_IncrementalDataset, CIFAR10_, BatchData
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

"""
单分支预测
"""
def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()  # AverageMeter可以记录当前的输出，累加到某个变量之中，然后根据需要可以打印出历史上的平均等信息
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    with paddle.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            data_time.update(time.time() - end)
            input = input.cuda()
            target = target.cuda()

            # output = model(input, 1, True)
            output = model(input, 10 / (epoch + 1), True)
            loss = criterion(output, target)  # 天坑！！原代码中是target.squeeze()
            prec1 = accuracy(output, target)
            losses.update(loss.item(), paddle.shape(input)[0])
            top1.update(prec1, paddle.shape(input)[0])
            loss_avg = losses.avg
            loss_avg = float(loss_avg[0])
            prec1_avg = top1.avg
            prec1_avg = float(prec1_avg[0])

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % 100 == 0 or (i + 1) == len(val_loader):
                print(
                    '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1:.4f}'.format(
                        batch=i + 1,
                        size=len(val_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=loss_avg,
                        top1=prec1_avg,
                    ))
    # print(prec1_avg)
    return (loss_avg, prec1_avg)


if __name__ == '__main__':
    print("你妈")
    args = argpar.get_args()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    valid_dataset = datasets.Cifar100(
        args.root,
        mode='test',
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    val_loader = paddle.io.DataLoader(
        # BatchData(val_x, val_y, input_transform_eval),
        valid_dataset,
        batch_size=1, shuffle=False,
        num_workers=args.workers)

    model = ForestNet('CIFAR100', 100, 256)
    model.load_dict(paddle.load('checkpoints2/GPU_hollylee_checkpoint_cifar100_tree_256_b.pth.tar')['state_dict'])

    # if paddle.cuda.is_available():
    #   model = model.cuda()
    """
    branch_params_list = list(map(id, model.branch_2.parameters())) + list(map(id, model.branch_3.parameters())) + list(
        map(id, model.branch_4.parameters()))

    # global_params = filter(lambda p: id(p) not in branch_params_list, model.parameters())
    branch_params = filter(lambda p: id(p) in branch_params_list, model.parameters())
    main_params = filter(lambda p: id(p) not in branch_params_list, model.parameters())
    for i in branch_params:
        i.optimize_attr['learning_rate'] = args.lr_global
    for i in main_params:
        i.optimize_attr['learning_rate'] = args.lr_global / 100
    """

    criterion = nn.NLLLoss()
    val_loss, prec1 = validate(val_loader, model, criterion, 200)
    print(val_loss)
    print(prec1)





