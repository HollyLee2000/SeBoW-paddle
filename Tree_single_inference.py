from LEARNTOBRANCH import LEARNTOBRANCH
import paddle.nn as nn
import paddle
import paddle.nn.initializer as init
from coder import *

"""
返回一个单分支前向预测的森林
"""


class ForestNet(LEARNTOBRANCH):
    def __init__(self, dataset, num_attributes, num_channels=64):
        super(ForestNet, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        fc_channel = 1
        self.dataset = dataset

        if dataset == 'CIFAR10':
            self.num_children = [1, 2, 4, 8, 8]
            self.num_in_channels = [3, num_channels // 4, num_channels // 2, num_channels, num_channels]
            self.num_out_channels = [num_channels // 4, num_channels // 2, num_channels, num_channels]
            self.cardinality = [1, 1, 1, 1, 1]
            # self.ds = [level1, level2, level3, level4]
            self.output_channels = [num_attributes] * 8
            self.classifier_nodes = [[1], [2], [4, 4], [8, 8, 8, 8]]
            self.fc_num = 8
        elif dataset == 'CIFAR100':
            # self.num_children = [1, 2, 4, 8, 8]
            # self.num_in_channels = [3, num_channels, num_channels, num_channels, num_channels]
            # self.num_out_channels = [num_channels, num_channels, num_channels, num_channels]
            # self.cardinality = [1, 1, 1, 1, 1]
            # self.output_channels = [num_attributes]*8
            # self.classifier_nodes = [[1], [2], [4, 4], [8, 8, 8, 8]]

            # self.num_children = [1, 2, 4, 8, 8]
            # self.num_in_channels = [3, num_channels, num_channels, num_channels, num_channels]
            # self.num_out_channels = [num_channels, num_channels, num_channels, num_channels]
            # self.cardinality = [1, 1, 1, 1, 1]
            # self.output_channels = [num_attributes] * 8
            # self.classifier_nodes = [[1], [2], [4, 4], [8]*4]
            self.fc_num = 4
            self.num_children = [1, 1, 1, 4, 4]
            self.ds = [level1, level2, level3, level4]
            self.num_in_channels = [3, num_channels // 4, num_channels // 2, num_channels, num_channels]
            self.num_out_channels = [num_channels // 4, num_channels // 2, num_channels, num_channels]
            self.cardinality = [1, 1, 1, 1, 1]
            self.output_channels = [num_attributes] * self.fc_num
            self.classifier_nodes = [[1], [1], [1], [4]]
        elif dataset == 'TINY-IMAGENET':
            self.fc_num = 8
            self.num_children = [1, 2, 4, 8, 8]
            self.num_in_channels = [3, num_channels // 4, num_channels // 2, num_channels, num_channels]
            self.num_out_channels = [num_channels // 4, num_channels // 2, num_channels, num_channels]
            self.cardinality = [1, 1, 1, 1, 1]
            self.output_channels = [num_attributes] * self.fc_num
            self.classifier_nodes = [[1], [2], [4, 4], [8] * 4]

        self.branches = []
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
                            nn.MaxPool2D(kernel_size=2)]))

                if layer != len(self.num_children) - 2 and self.classifier_nodes[layer + 1][i] > 1:
                    setattr(self, 'router{}_{}'.format(str(layer + 2), str(i)),
                            nn.Sequential(*[
                                nn.Conv2D(self.num_in_channels[layer + 1], num_channels // 2, kernel_size=3, padding=1),
                                nn.ReLU(True),
                                nn.Conv2D(num_channels // 2, num_channels // 2, kernel_size=3, padding=1),
                                nn.ReLU(True),
                                nn.AdaptiveAvgPool2D((3, 3))
                            ]))

                    setattr(self, 'router_classifier{}_{}'.format(str(layer + 2), str(i)),
                            nn.Sequential(*[
                                nn.Linear(9 * num_channels // 2, self.classifier_nodes[layer + 1][i]),
                                nn.Softmax()
                            ]))

        for i in range(self.fc_num):
            if self.output_channels[i] > 1:
                setattr(self, 'fc1_' + str(i),
                        nn.Sequential(*[
                            nn.Linear(fc_channel * num_channels, self.output_channels[i]),
                            nn.Softmax()
                        ]))

        self.num_attributes = num_attributes
        self._initialize_weights()

    def forward(self, x, t=10, training=True):
        bs = x.shape[0]

        xs = []  # 用于生成下一层的张量，single-path状态下则只有一个
        x_branches = [x]  # 当前层的输入张量，single-path状态下则只有一个
        pro = paddle.ones([bs, 1]).cuda()  # 当前层每个learner的输出，single-path状态下则只有一个
        pros = []
        tool = 0  # 当前概率最大的孩子节点标号
        for layer in range(len(self.num_children) - 1):
            layer_child = self.num_children[layer]
            conv = getattr(self, 'conv{}_{}'.format(str(layer + 2), str(tool)))  # 获得概率最大的learner卷积结果
            # print(layer, i)
            after_conv = conv(x_branches[0])
            xs.append(after_conv)  # 生成传到下一层learner的输出
            if layer != len(self.num_children) - 2 and self.classifier_nodes[layer + 1][
                tool] > 1:  # 当前learner的后代如果不止一个才需要sender
                router = getattr(self, 'router{}_{}'.format(str(layer + 2), str(tool)))
                classifier = getattr(self, 'router_classifier{}_{}'.format(str(layer + 2), str(tool)))
                pro = router(after_conv)
                pro = paddle.reshape(pro, [bs, -1])
                pro = classifier(pro)
                pro = paddle.clip(pro, 0.001, 0.999)  # paddle.clip==pytorch.clamp
                # pro = (pro >= torch.max(pro, 1)[0].view(bs, -1)).float() * pro
                # pro = pro * (paddle.reshape(pre_pro[i], [bs, -1]))
                # pro = paddle.clip(pro, 0.001, 0.999)
                pros.append(pro[0])  # 存储learner输出经过sender处理后的结果，因为预测时是单样本输入，取CHW出来
            else:
                pros.append(paddle.to_tensor([1]))
            # print(pros)

            if layer != len(self.num_children) - 2:
                now_max = 0  # 表示当前最大概率
                for i in range(self.num_children[layer + 1]):  # child
                    # print(pros[0][i])
                    pro = pros[0][i]
                    if pro > now_max:
                        now_max = pro
                        tool = i  # 更新概率最大的孩子节点标号
                x_branches = [xs[0]]
                xs = []
                pros = []

        outputs = 0
        tx = xs[0]
        tx = self.avgpool(tx)
        # outputs+=pro
        tx = paddle.reshape(tx, [paddle.shape(x)[0], -1])
        if self.output_channels[tool] > 1:
            fc1 = getattr(self, 'fc1_' + str(tool))
            out = fc1(tx)
            out = paddle.clip(out, 0.001, 0.999)
            # pro = torch.clamp(pro, 0.001, 0.999)
            outputs = out

        # print(torch.sum(outputs, 1))
        outputs = paddle.log(outputs)

        return outputs


"""
from thop import profile
from thop import clever_format
from torchvision import models

# model = ForestNet('TINY-IMAGENET', num_attributes=200, num_channels=256).cuda()
# model = ForestNet('CIFAR100', 100, num_channels=288).cuda()
model = ForestNet('CIFAR100', 100, num_channels=288)
input = torch.randn(1, 3, 64, 64)
# macs, params = profile(model, inputs=(input,))
# macs, params = clever_format([macs, params], "%.3f")
# print(macs, params)
out = model.forward(input)
print('???output:', out.shape)

# forest  279.831 4.742M
# forest imitate adaptive neural network num_channel=64 1.439M
# forest .... num_channel=96  2.879M
"""
