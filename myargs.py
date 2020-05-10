import argparse

parser = argparse.ArgumentParser()

# ======= Model parameters =======

parser.add_argument('--model_name', default='resnet18', type=str,
                    help='pretrained model name')
parser.add_argument('--num_electrodes', default=20, type=int,
                    help='number of electrodes used for input')
parser.add_argument('--workers', default=8, type=int,
                    help='number of workers to use for GenerateIterator')

parser.add_argument('--lr', default=0.001, type=float,
                    help='learning rate')
parser.add_argument('--weight_decay', default=0.0001, type=float,
                    help='weight decay/weights regularizer for sgd')
parser.add_argument('--beta1', default=0.9, type=float,
                    help='momentum for sgd, beta1 for adam')
parser.add_argument('--beta2', default=0.999, type=float,
                    help='momentum for sgd, beta1 for adam')

parser.add_argument('--num_epochs', default=100, type=int,
                    help='epochs to train for')
parser.add_argument('--start_epoch', default=1, type=int,
                    help='epoch to start training. useful if continue from a checkpoint')
parser.add_argument('--early_break', default=-1, type=int,
                    help='for debugging. only train on this amount of batches. -1 for normal training')
parser.add_argument('--pretrain_epoch', default=1, type=int,
                    help='epoch to start training. useful if continue from a checkpoint')

parser.add_argument('--batch_size', default=64, type=int,
                    help='input batch size')

parser.add_argument('--window_len', default=30, type=int,
                    help='length of window to cut out to perform STFT on')
parser.add_argument('--label_0_overlap', default=0, type=int,
                    help='overlap between 2 adjacent windows if they are label 0')
parser.add_argument('--label_1_overlap', default=28, type=int,
                    help='overlap between 2 adjacent windows if they are label 1')
parser.add_argument('--seiz_sens', default=0.3, type=float,
                    help='percent of window that must be positive to label it seizure')


args = parser.parse_args()
