import argparse

parser = argparse.ArgumentParser()

# ======= Model parameters =======

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
parser.add_argument('--pretrain_epoch', default=1, type=int,
                    help='epoch to start training. useful if continue from a checkpoint')

parser.add_argument('--batch_size', default=16, type=int,
                    help='input batch size')
parser.add_argument('--window_len', default=30, type=int,
                    help='length of window to cut out to perform STFT on')
parser.add_argument('--overlap', default=15, type=int,
                    help='overlap between 2 adjacent windows')
parser.add_argument('--seiz_sens', default=0.3, type=float,
                    help='percent of window that must be positive to label it seizure')


args = parser.parse_args()