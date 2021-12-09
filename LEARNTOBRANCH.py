import paddle.nn as nn
import paddle
import math
import numpy as np
import paddle.nn.initializer as init  # pytorch为nn.init

"""
森林结构基类
"""
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
            m.weight_attr = init.Normal(0, 0.01)  # 随机正态（高斯）分布初始化函数
            m.bias_attr = init.Constant(0)


def gumbel_softmax(s, t=1):
    """
    print(paddle.pow(paddle.to_tensor([math.e]), s / t) / paddle.reshape(
        paddle.sum(paddle.pow(paddle.to_tensor([math.e]), s / t), 1), [-1, 1]))
    """
    tool1 = paddle.to_tensor(np.power(math.e, np.array(s / t)))
    tool2 = paddle.reshape(paddle.sum(paddle.to_tensor(np.power(math.e, np.array(s / t))), 1), [-1, 1])
    return tool1 / tool2


class LEARNTOBRANCH(nn.Layer):
    def __init__(self):
        super(LEARNTOBRANCH, self).__init__()

    def _initialize_weights(self):
        init_params(self)

    def branching_op(self, branch, par, chi, t, training=True):
        ''' gumbel-softmax
        Parameters: branch,  parent number,  children number,  temperature'''
        d = paddle.zeros([chi, par])  # zeros和pytorch的用法也不同，需要声明dtype
        pro = branch[0]
        pro = gumbel_softmax(pro, 0.5)
        # print("1: ", pro)
        # wuyuzi = paddle.to_tensor([float(pro[0]), 1])  # paddle.max只能求tensor中的最大值  要先转tensor,pro已经是tensor,先取第一个元素
        ind = paddle.max(pro, 1)  # paddle.max只能求tensor中的最大值,不会返回最大值所在下标
        for i in range(chi):
            if training:
                d[i] += paddle.log(pro[i])  # 这是论文中加的那一点噪音
            else:
                print("出大问题")
                # d[i][ind[i]] += 1
        if training:
            d = gumbel_softmax(d, t)
        # print("2: ", d)
        return d


class LEARNTOBRANCH_Deep(LEARNTOBRANCH):
    def __init__(self, dataset, num_attributes=95, loss_method='nce', num_channels=160):
        super(LEARNTOBRANCH_Deep, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        # self.conv1_0 = nn.Sequential(*[nn.Conv2d(3, 64, kernel_size=7), nn.BatchNorm2d(64), nn.ReLU(True),
        # nn.MaxPool2d(kernel_size=2, stride=2)])
        fc_channel = 1
        self.dataset = dataset

        if dataset == 'CIFAR10':
            # self.num_children = [2, 4, 8, num_attributes]
            # self.num_in_channels = [3, 64, 64, 32]
            # self.num_out_channels = [64, 64, 64]
            self.num_children = [1, 1, 2, 4, num_attributes]
            self.num_in_channels = [3, 64, 64, 64, 32]
            self.num_out_channels = [64, 64, 64, 64]
        elif dataset == 'CIFAR100':
            self.num_children = [1, 2, 4, 8, num_attributes]
            self.num_in_channels = [3, num_channels, num_channels, num_channels, num_channels]
            self.num_out_channels = [num_channels, num_channels, num_channels, num_channels]
        else:  # CelebA AZFT
            self.num_children = [2, 4, 8, 16, num_attributes]
            self.num_in_channels = [64, 64, 64, 64, 32]
            self.num_out_channels = [64, 64, 64, 64]

        self.branches = []
        self.loss_method = loss_method
        if loss_method == 'nce':
            output_channel = 2
        else:
            output_channel = 1

        for layer in range(len(self.num_children) - 1):
            layer_child = self.num_children[layer]

            for i in range(layer_child):  # block
                setattr(self, 'conv{}_{}'.format(str(layer + 2), str(i)),
                        nn.Sequential(*[
                            nn.Conv2D(self.num_in_channels[layer], self.num_out_channels[layer], kernel_size=3,
                                      padding=1),
                            nn.BatchNorm2D(self.num_out_channels[layer]),
                            nn.ReLU(True),
                            nn.Conv2D(self.num_out_channels[layer], self.num_in_channels[layer + 1], kernel_size=3,
                                      padding=1),
                            nn.BatchNorm2D(self.num_in_channels[layer + 1]),
                            nn.ReLU(True),
                            nn.MaxPool2D(kernel_size=2, stride=2)]))

            # probability matrix
            """
            这里两个框架差的有点多  不知道会不会出问题
            """
            setattr(self, 'branch_{}'.format(str(layer + 2)),
                    nn.ParameterList([paddle.create_parameter(
                        shape=[self.num_children[layer + 1], self.num_children[layer]], dtype="float32",
                        default_initializer=init.Uniform(low=0, high=1))]))
            self.branches.append(getattr(self, 'branch_{}'.format(str(layer + 2))))

        for i in range(num_attributes):
            # setattr(self, 'fc1_' + str(i), nn.Sequential(*[nn.Linear(fc_channel * 32, 128), nn.ReLU(True),
            # nn.Dropout()]))
            setattr(self, 'fc2_' + str(i), nn.Sequential(*[nn.Linear(fc_channel * num_channels, output_channel)]))

        self.num_attributes = num_attributes
        self._initialize_weights()

    def forward(self, x, t=10, training=True):
        # if self.dataset != 'CIFAR10' and self.dataset != 'CIFAR100':
        #    x = self.conv1_0(x)

        xs = []  # store the output from previous layer
        x_branches = [x]  # next level input

        for layer in range(len(self.num_children) - 1):
            layer_child = self.num_children[layer]
            for i in range(layer_child):  # block
                conv = getattr(self, 'conv{}_{}'.format(str(layer + 2), str(i)))
                xs.append(conv(x_branches[i]))
            x_branches = []
            d = self.branching_op(self.branches[layer], layer_child, self.num_children[layer + 1], t, training)
            for i in range(self.num_children[layer + 1]):
                for j in range(layer_child):
                    if j == 0:
                        x_branch = xs[j] * d[i][j]
                    """
                        else:
                            x_branch += xs[j] * d[i][j]
                    """
                x_branches.append(x_branch)
            xs = []

        outputs = []
        for i in range(self.num_attributes):
            tx = x_branches[i]
            tx = self.avgpool(tx)
            tx = paddle.reshape(tx, [paddle.shape(tx)[0], -1])  # pytorch通常为属性自带方法，而paddle为外部调用
            # fc1 = getattr(self, 'fc1_' + str(i))
            fc2 = getattr(self, 'fc2_' + str(i))
            outputs.append(fc2(tx))

        if self.loss_method == 'ce':
            outputs = paddle.concat(outputs, 1)  # 代替pytorch的cat

        return outputs


from thop import profile
from thop import clever_format

model = LEARNTOBRANCH_Deep('CIFAR100', 100, 'ce', 160)
input = paddle.randn(shape=[1, 3, 32, 32])
output = model.forward(input)
# print('output_shape:', output.shape)
# print('output:', output)
