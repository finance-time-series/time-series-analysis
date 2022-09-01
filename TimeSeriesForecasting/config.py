import argparse

parser = argparse.ArgumentParser()

# CPU / GPU setting
parser.add_argument('--device', type=str, default='2')
parser.add_argument('--use_cuda', type=str, default='True')
parser.add_argument('--device_num', type=str, default=0)

# training hyperparameter
parser.add_argument('--model', type=str, default='ncde')

parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--data', type=str, default='bank')
parser.add_argument('--type', type=str, default='elec')

parser.add_argument('--train_seq', type=int, default=30)
parser.add_argument('--y_seq', type=int, default=1)


# model hyperparameter
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--hh_size', type=int, default=128)
parser.add_argument('--n_layers', type=int, default=1)

def get_config():
    return parser.parse_args()
