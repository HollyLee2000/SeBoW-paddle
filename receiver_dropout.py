import paddle
from ForestModel import ForestNet

"""
打印receiver输出，选择唯一父母节点
"""
model = ForestNet('CIFAR100', 100, 256)
model.set_state_dict(paddle.load('checkpoints2/model_best_cifar100_forest_256_b.pth.tar')['state_dict'])
print('# forest parameters:', sum(param.numel() for param in model.parameters()))
for i in model.state_dict():
    if str(i).find("branch") != -1:
        print(i, ": ", model.state_dict()[str(i)])
