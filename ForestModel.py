from LEARNTOBRANCH import LEARNTOBRANCH
import paddle.nn as nn
import paddle
import paddle.nn.initializer as init
from coder import *

"""
森林结构
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
            self.fc_num = 8
            self.num_children = [1, 2, 4, 8, 8]
            self.num_in_channels = [3, num_channels // 4, num_channels // 2, num_channels, num_channels]
            self.num_out_channels = [num_channels // 4, num_channels // 2, num_channels, num_channels]
            self.cardinality = [1, 1, 1, 1, 1]
            self.output_channels = [num_attributes] * self.fc_num
            self.classifier_nodes = [[1], [2], [4, 4], [8] * 4]
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

            if layer < len(self.num_children) - 2:
                setattr(self, 'branch_{}'.format(str(layer + 2)),
                        nn.ParameterList([paddle.create_parameter(
                            shape=[self.num_children[layer + 1], self.num_children[layer]], dtype="float32",
                            default_initializer=init.Uniform(low=0, high=1))]))
                self.branches.append(getattr(self, 'branch_{}'.format(str(layer + 2))))

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

        xs = []  # store the output from previous layer
        x_branches = [x]  # next level input
        pro = paddle.ones([bs, 1])
        pre_pro = [pro]
        pros = []

        for layer in range(len(self.num_children) - 1):
            layer_child = self.num_children[layer]
            for i in range(layer_child):  # block
                conv = getattr(self, 'conv{}_{}'.format(str(layer + 2), str(i)))
                # print(layer, i)
                after_conv = conv(x_branches[i])
                xs.append(after_conv)
                if layer != len(self.num_children) - 2:
                    if self.classifier_nodes[layer + 1][i] > 1:
                        router = getattr(self, 'router{}_{}'.format(str(layer + 2), str(i)))
                        classifier = getattr(self, 'router_classifier{}_{}'.format(str(layer + 2), str(i)))
                        pro = router(after_conv)
                        pro = paddle.reshape(pro, [bs, -1])
                        pro = classifier(pro)
                        pro = paddle.clip(pro, 0.001, 0.999)  # paddle.clip==pytorch.clamp
                        # pro = (pro >= torch.max(pro, 1)[0].view(bs, -1)).float() * pro
                        pro = pro * (paddle.reshape(pre_pro[i], [bs, -1]))
                        pro = paddle.clip(pro, 0.001, 0.999)
                        pros.append(pro)
                    else:
                        pros.append(paddle.ones([bs, 1]) * (paddle.reshape(pre_pro[i], [bs, -1])))

            if layer != len(self.num_children) - 2:
                x_branches = []
                pre_pro = []

                d = self.branching_op(self.branches[layer], layer_child, self.num_children[layer + 1], t, training)

                indexes = [0] * layer_child

                for i in range(self.num_children[layer + 1]):  # child
                    pro = 0
                    x_branch = 0
                    for j in range(layer_child):  # par
                        if j == 0:
                            x_branch = xs[j] * d[i][j]  # xs[j]是上一层的输出
                        else:
                            x_branch += xs[j] * d[i][j]
                        if layer != len(self.num_children) - 2:
                            pro += pros[j][:, indexes[j]]
                            indexes[j] += 1
                    x_branches.append(x_branch)  # 为每一个下一层的lerner,生成一个输入
                    if layer != len(self.num_children) - 2:
                        pre_pro.append(pro)
                xs = []
                pros = []

        outputs = 0
        for i in range(self.fc_num):
            tx = xs[i]
            pro = paddle.reshape(pre_pro[i], [bs, -1])
            tx = self.avgpool(tx)
            pro = paddle.clip(pro, 0.001, 0.999)
            # outputs+=pro
            tx = paddle.reshape(tx, [paddle.shape(x)[0], -1])
            if self.output_channels[i] > 1:
                fc1 = getattr(self, 'fc1_' + str(i))
                out = fc1(tx)
                out = paddle.clip(out, 0.001, 0.999)
                pro = out * pro
                # pro = torch.clamp(pro, 0.001, 0.999)
            outputs += pro

        # print(torch.sum(outputs, 1))
        outputs = paddle.log(outputs)

        return outputs


from torchvision import models

# model = ForestNet('TINY-IMAGENET', num_attributes=200, num_channels=256).cuda()
# model = ForestNet('CIFAR100', 100, num_channels=288).cuda()
model = ForestNet('CIFAR100', 100, num_channels=288)
input = paddle.randn(shape=[1, 3, 64, 64])
# macs, params = profile(model, inputs=(input,))
# macs, params = clever_format([macs, params], "%.3f")
# print(macs, params)
out = model.forward(input)
# print('output:', out)

# forest  279.831 4.742M
# forest imitate adaptive neural network num_channel=64 1.439M
# forest .... num_channel=96  2.879M
