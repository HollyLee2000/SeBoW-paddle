import argparse

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--num_classes', type=int)
parser.add_argument('--arch', default='deep', type=str)
parser.add_argument('--loss_method', type=str, default='nce',
                    help='ce or nce')  # CrossEntropy  N_classes x CrossEntropy
parser.add_argument('--root', type=str, default='./datasets/cifar-100-python.tar.gz', help='Dataset location')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number')
parser.add_argument('--train_batch', default=128, type=int, help='train batchsize')
parser.add_argument('--test_batch', default=128, type=int, help='test batchsize')
parser.add_argument('--lr_global', '--learning_rate_global', default=0.1, type=float,
                    help='initial global learning rate')
parser.add_argument('--lr_branch', '--learning_rate_branch', default=1e-2, type=float,
                    help='initial branch learning rate')
parser.add_argument('--lr_stage2', default=0.05, type=float, help='initial learning rate for stage 2 training')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--weight_decay_stage2', default=5e-4, type=float, help='weight decay for stage 2')
parser.add_argument('--checkpoint', default='./checkpoints2', type=str, help='path to save checkpoint')
parser.add_argument('--t', default=10, type=int, help='inital temperature')
parser.add_argument('--resume', type=str, help='location of the sampled pretrained model')
args = parser.parse_args()


def get_args():
    return args
