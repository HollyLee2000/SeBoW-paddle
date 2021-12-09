import paddle
import paddle.vision.transforms as transforms
import argpar
import os
import copy
import paddle.nn as nn
from Forest_cal_sender import ForestNet
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
利用sender的输出淘汰节点
"""
def validate(val_loader, model, epoch):
    model.eval()
    re = dict()
    print("你妈")
    with paddle.no_grad():
        for i, (input, __) in enumerate(val_loader):
            input = input.cuda()
            _, pros = model(input, 10 / (epoch + 1), True)
            for j in pros:
                re[j] = paddle.zeros(pros[j][0].shape).cuda()
            break
    count = 0
    with paddle.no_grad():
        for i, (input, __) in enumerate(val_loader):
            input = input.cuda()
            _, pros = model(input, 10 / (epoch + 1), True)
            for j in pros:
                re[j] += pros[j][0]
            count += 1
            print(count, "OJBK")
        for k in re:
            re[k] /= count
    print(count)
    for k in re.items():
        print(k)


if __name__ == '__main__':
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
    model.set_state_dict(paddle.load('checkpoints2/model_best_cifar100_forest_256_b.pth.tar')['state_dict'])

    optimizer = paddle.optimizer.Momentum(parameters=model.parameters(),
                                          learning_rate=args.lr_global,
                                          # args.lr_stage2/10,
                                          momentum=args.momentum,
                                          weight_decay=1e-4)

    validate(val_loader, model, 200)
