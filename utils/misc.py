import errno
import os
import paddle
import paddle.nn as nn
import paddle.nn.initializer as init  # pytorch为nn.init

__all__ = ['get_mean_and_std', 'init_params', 'mkdir_p', 'AverageMeter']


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    # 对应torch.utils.data.DataLoader
    dataloader = trainloader = paddle.io.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = paddle.zeros(shape=[3])
    std = paddle.zeros(shape=[3])
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    """
    Init layer parameters.
    写一个递归函数作为模块迭代器，用于初始化参数
    """
    for m in net.children():
        if isinstance(m, nn.Sequential):
            init_params(m)
        elif isinstance(m, nn.Conv2D):
            m.weight_attr = init.KaimingNormal(fan_in=None)  # 该接口实现Kaiming正态分布方式的权重初始化
            if m.bias is not None:
                m.bias_attr = init.Constant(0)  # 该接口为常量初始化函数，用于权重初始化，通过输入的value值初始化输入变量,和pytorch相比少了一个参数tensor，转为直接赋值
        elif isinstance(m, nn.BatchNorm2D):
            m.weight_attr = init.Constant(1)
            m.bias_attr = init.Constant(0)
        elif isinstance(m, nn.Linear):
            m.weight_attr = init.Normal(std=1e-3)  # 随机正态（高斯）分布初始化函数
            if m.bias is not None:
                m.bias_attr = init.Constant(0)


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


# AverageMeter可以记录当前的输出，累加到某个变量之中，然后根据需要可以打印出历史上的平均等信息
class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

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
