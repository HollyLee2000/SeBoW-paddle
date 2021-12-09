import paddle
from Forest_to_tree import ForestNet

"""
打印模型参数和内容
"""
model = ForestNet('CIFAR100', 100, 256)
model.load_dict(paddle.load('checkpoints2/GPU_hollylee_checkpoint_cifar100_tree_256_b.pth.tar')['state_dict'])
print('# forest parameters:', sum(param.numel() for param in model.parameters()))
print("打印模型所有层名称：")
for i in model.state_dict():
    print(i)
print("打印模型所有层名称：")
tool = paddle.load('checkpoints2/GPU_hollylee_checkpoint_cifar100_tree_256_b.pth.tar')['state_dict']
for i in tool:
    print(i)
