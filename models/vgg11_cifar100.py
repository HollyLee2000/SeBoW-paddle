# vgg11 for cifar100
import paddle
import paddle.nn as nn
import paddle.nn.initializer as init  # pytorch为nn.init

"""
用paddlepaddle实现vgg11
paddlepaddle的Layer代替了pytorch框架下的网络基类Module
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
            m.weight_attr = init.Normal(0)
            m.bias_attr = init.Constant(0)


class VGG(nn.Layer):
    """
    VGG model
    """

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features  # 这里的features就是经过了VGG网络的make_layer函数处理的512通道特征图
        """
            classifier是将vgg最后产生的特征图变成一维数据后再进行分类的网络
            cifar100数据集输出100类
        """
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 100),
        )
        init_params(self)
        """
            paddlepaddle关于权重初始化方面的坑：
            1.pytorch有模块迭代器modules而paddlepaddle没有，所以只有通过children等方法去找含有卷积操作的模块，分别初始化权重
            2.paddlepaddle不能调用除了weight的所有conv2D属性
            解决方法：
            1.手动挖出所需的网络属性，用全局变量保存(很麻烦，而且还是有很多问题难以解决)
            2.在建设VGG时同时初始化偏置和权重(最终采用的是这个,但是初始化方法变了)
        """

    def forward(self, x):
        print("1", x)
        x = self.features(x)
        print("2", x)
        # pytorch中有tensor.view()方法可以调整tensor的形状，而paddlepaddle没有，这里借助了reshape来帮忙
        x = paddle.reshape(x, [paddle.shape(x)[0], -1])
        print("3", x)
        x = self.classifier(x)
        print("4", x)
        return x


def make_layers(this_cfg, batch_norm=True):
    layers = []
    in_channels = 3
    for v in this_cfg:
        if v == 'M':
            layers += [nn.MaxPool2D(kernel_size=2, stride=2)]
        else:
            """
            初始化偏置和权重，和pytorch源码效果或许会不一样
            这里是Kaiming正态分布方式的权重初始化，偏置参数采用全0初始化方式
            """
            conv2d = nn.Conv2D(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2D(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A'], True))


if __name__ == '__main__':
    model = vgg11()
    """
    randn返回符合标准正态分布（均值为0，标准差为1的正态随机分布）的随机Tensor,但是torch和paddle的参数形式不同，这也导致不能
    使用profile函数(传入tensor会报错)
    """
    input = paddle.randn(shape=[4, 3, 32, 32])
    output = model.forward(input)
    print('output:', output.shape)
