import argparse

def get_options(parser=argparse.ArgumentParser()):
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--continue_training', action="store_true", default=False)
    parser.add_argument('--load_epoch_num', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=1)
    opt = parser.parse_args()
    return opt
