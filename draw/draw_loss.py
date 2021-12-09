from matplotlib import pyplot as plt

filename = 'paddle.txt'
filename2 = 'torch.txt'
Val_loss_paddle, Val_acc_paddle, Val_loss_torch, Val_acc_torch, epoch = [], [], [], [], []
now_epoch = 0
with open(filename, 'r') as f:
    for line in f.readlines():
        temp = line.split('\t')
        Val_acc_paddle.append(float(temp[2]))

with open(filename2, 'r') as f2:
    for line in f2.readlines():
        temp = line.split('\t')
        Val_acc_torch.append(float(temp[2]))
        epoch.append(now_epoch)
        now_epoch += 1

plt.figure()
plt.plot(epoch, Val_acc_paddle, 'red', label='Val_loss_paddle')
plt.plot(epoch, Val_acc_torch, 'blue', label='Val_loss_torch')
plt.legend()
plt.show()

"""
fig = plt.figure(figsize=(10, 5))  # 创建绘图窗口，并设置窗口大小
# 画第一张图
ax1 = fig.add_subplot(211)  # 将画面分割为2行1列选第一个
ax1.plot(epoch, Val_loss_paddle, 'red', label='Val_loss_paddle')  # 画dis-loss的值，颜色红
ax1.legend(loc='upper right')  # 绘制图例，plot()中的label值
ax1.set_xlabel('epoch')  # 设置X轴名称
ax1.set_ylabel('Val_loss')  # 设置Y轴名称
# 画第二张图
ax2 = fig.add_subplot(212)  # 将画面分割为2行1列选第二个
ax2.plot(epoch, Val_loss_torch, 'blue', label='Val_loss_torch')  # 画gan-loss的值，颜色蓝
ax2.legend(loc='upper right')  # loc为图例位置，设置在右上方，（右下方为lower right）
ax2.set_xlabel('epoch')
ax2.set_ylabel('Val_loss')
plt.show()  # 显示绘制的图
"""



