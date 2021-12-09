import paddle
import paddle.vision.transforms as transforms
import argpar
import os
import copy
import paddle.nn as nn
from Forest_to_tree import ForestNet
from tensorboardX import SummaryWriter
import time
from utils.logger import Logger, savefig
from utils.misc import AverageMeter
from utils.util import accuracy, save_checkpoint, adjust_learning_rate2  # , accuracy_stage2
import paddle.vision.datasets as datasets
from dataset import CIFAR100_IncrementalDataset, CIFAR10_, BatchData
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

"""
树结构的训练
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

            loss = criterion(output, target.squeeze())
            prec1 = accuracy(output, target.squeeze())
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


def train(train_loader, model, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        # output = model(input, 1, True)
        output = model(input, 10 / (epoch + 1), True)

        loss = criterion(output, target.squeeze())
        # print(loss)
        # exit(-1)

        prec1 = accuracy(output, target.squeeze())
        losses.update(loss.item(), paddle.shape(input)[0])
        top1.update(prec1, paddle.shape(input)[0])
        loss_avg = losses.avg
        loss_avg = float(loss_avg[0])
        prec1_avg = top1.avg
        prec1_avg = float(prec1_avg[0])

        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % 100 == 0 or (i + 1) == len(train_loader):
            print('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1:.4f}'.format(
                batch=i + 1,
                size=len(train_loader),
                data=data_time.avg,
                bt=batch_time.avg,
                loss=loss_avg,
                top1=prec1_avg,
            ))
    return (loss_avg, prec1_avg)


if __name__ == '__main__':
    args = argpar.get_args()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.Cifar100(
        args.root,
        mode='train',
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ]))

    valid_dataset = datasets.Cifar100(
        args.root,
        mode='test',
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = paddle.io.DataLoader(
        # BatchData(train_x, train_y, input_transform),
        train_dataset,
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers)

    val_loader = paddle.io.DataLoader(
        # BatchData(val_x, val_y, input_transform_eval),
        valid_dataset,
        batch_size=128, shuffle=False,
        num_workers=args.workers)

    model = ForestNet('CIFAR100', 100, 256)

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
    optimizer = paddle.optimizer.Momentum(parameters=model.parameters(),
                                          learning_rate=args.lr_global,
                                          # args.lr_stage2/10,
                                          momentum=args.momentum,
                                          weight_decay=1e-4)

    criterion = nn.NLLLoss()
    best_prec1 = 0
    writer = SummaryWriter(os.path.join(args.checkpoint, 'holly_logs_cifar100_tree_256_b'))
    logger = Logger(os.path.join(args.checkpoint, 'holly_log_cifar100_tree_256_b.txt'), title='cifar100_tree')
    logger.set_names(['Global Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    for epoch in range(args.start_epoch, args.epochs):
        global_lr = adjust_learning_rate2(args, optimizer, epoch)

        print('\nEpoch: [%d | %d] global_LR: %f' % (epoch + 1, args.epochs, global_lr))

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, writer)

        val_loss, prec1 = validate(val_loader, model, criterion, epoch)

        logger.append([global_lr, train_loss, val_loss, train_acc, prec1])

        writer.add_scalar('global_learning rate', global_lr, epoch + 1)
        writer.add_scalar('train_loss', train_loss, epoch + 1)
        writer.add_scalar('validation_loss', val_loss, epoch + 1)
        writer.add_scalar('train accuracy', train_acc, epoch + 1)
        writer.add_scalar('validation accuracy', prec1, epoch + 1)
        # if epoch>30:#:best_prec1>85 and prec1<80:
        #     break
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint, filename='hollylee_checkpoint_cifar100_tree_256_b.pth.tar',
            best_filename='model_best_cifar100_tree_256_b.pth.tar')

    print('Best accuracy:')
    print(best_prec1)

    logger.close()
    writer.close()





