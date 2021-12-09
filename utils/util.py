import os
import paddle
import shutil


#
# def accuracy(output, target):
#     with torch.no_grad():
#         batch_size = target.size(0)
#         _, pred = output.max(1)
#         correct = pred.eq(target).sum().item()
#         res = correct*100.0/batch_size
#         return res

def accuracy(output, target):
    with paddle.no_grad():
        # print(paddle.shape(target)[0])
        batch_size = paddle.shape(target)[0]
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.equal(paddle.reshape(target, [1, -1]))
        correct = paddle.to_tensor(paddle.reshape(correct, [-1]), dtype='float32').sum(0)  # 没有投射为float类型
        res = paddle.multiply(correct, paddle.to_tensor(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint_stage2.pth.tar',
                    best_filename='model_best_stage2.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    paddle.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, best_filename))


def adjust_learning_rate2(args, optimizer, epoch):
    global_lr = optimizer.get_lr()

    if 31 > epoch > 0 and epoch % 30 == 0:
        global_lr = global_lr / 4

    if epoch >= 50 and (epoch - 30) % 20 == 0:
        global_lr = global_lr / 4

    optimizer.set_lr(global_lr)

    return global_lr


def adjust_learning_rate3(args, optimizer, epoch):
    global_lr = optimizer.param_groups[0]['lr']

    if 31 > epoch > 0 and epoch % 30 == 0:
        global_lr = global_lr / 4

    if epoch >= 50 and (epoch - 30) % 20 == 0:
        global_lr = global_lr / 4

    optimizer.param_groups[0]['lr'] = global_lr

    return global_lr


def adjust_learning_rate4(args, optimizer, epoch):
    global_lr = optimizer.param_groups[0]['lr']

    if 50 > epoch > 0 and epoch % 50 == 0:
        global_lr = global_lr / 4

    if epoch >= 50 and (epoch - 30) % 20 == 0:
        global_lr = global_lr / 4

    optimizer.param_groups[0]['lr'] = global_lr

    return global_lr


def adjust_learning_rate5(args, optimizer, epoch):
    global_lr = optimizer.param_groups[0]['lr']

    if epoch >= 40 and (epoch - 40) % 20 == 0:
        global_lr = global_lr / 4

    optimizer.param_groups[0]['lr'] = global_lr

    return global_lr


def adjust_learning_ratesave(args, optimizer, epoch):
    global_lr = optimizer.param_groups[0]['lr']

    if epoch >= 13 and (epoch - 13) % 20 == 0:
        global_lr = global_lr / 4

    optimizer.param_groups[0]['lr'] = global_lr

    return global_lr


def adjust_learning_ratesave2(args, optimizer, epoch):
    global_lr = optimizer.param_groups[0]['lr']

    if epoch >= 21 and (epoch - 21) % 20 == 0:
        global_lr = global_lr / 4

    optimizer.param_groups[0]['lr'] = global_lr

    return global_lr


def adjust_learning_rate(args, optimizer, epoch):
    global_lr = optimizer.param_groups[0]['lr']
    branch_lr = optimizer.param_groups[1]['lr']

    # warmup without update branch probabilities
    if epoch == 0:
        global_lr = args.lr_global / 4
        branch_lr = 0
    elif epoch == 1:
        global_lr = args.lr_global / 2
        branch_lr = 0
    elif epoch == 2:
        global_lr = args.lr_global
        branch_lr = args.lr_branch

    # exponential decay 2.4
    elif epoch % 2 == 0:
        global_lr = args.lr_global * (1 - epoch / int(args.epochs * 1.03093))
        branch_lr = args.lr_branch * (1 - epoch / int(args.epochs * 1.03093))

    optimizer.param_groups[0]['lr'] = global_lr
    optimizer.param_groups[1]['lr'] = branch_lr
    return global_lr, branch_lr


def adjust_learning_rate_stage2(args, optimizer, epoch):
    global_lr = optimizer.param_groups[0]['lr']

    if epoch >= 10 and epoch % 10 == 0:
        global_lr = global_lr / 2

    optimizer.param_groups[0]['lr'] = global_lr
    return global_lr
