import paddle
import torch
import numpy as np
import ForestModel
import argpar

exp = torch.tensor([[-0.1575, -1.9261],
                    [-0.6005, -0.7952],
                    [-0.5653, -0.8398],
                    [-0.3435, -1.2355]])
print(torch.pow(2.33, exp))
exp2 = paddle.to_tensor([[0.1575, 1.9261],
                         [-0.6005, 0.7952],
                         [-0.5653, -0.8398],
                         [-0.3435, -1.2355]])
print(paddle.pow(paddle.to_tensor([2.33]), exp2))
tool = [[-0.1575, -1.9261],
        [-0.6005, -0.7952],
        [-0.5653, -0.8398],
        [-0.3435, -1.2355]]
print(np.power(2.33, tool))

model = ForestModel.ForestNet('CIFAR100', 100, 256)
args = argpar.get_args()
optimizer = paddle.optimizer.SGD(parameters=model.parameters(),
                                 learning_rate=args.lr_global,
                                 # args.lr_stage2/10,
                                 # momentum=args.momentum,
                                 weight_decay=1e-4)
for i in model.parameters():
    i.optimize_attr['learning_rate'] /= 4
