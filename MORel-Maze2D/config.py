import argparse

def get_options(parser=argparse.ArgumentParser()):
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--continue_training', action="store_true", default=False)
    parser.add_argument('--load_epoch_num', type=int, default=0)
    parser.add_argument('--continue_training_policy', action="store_true", default=False)
    parser.add_argument('--load_epoch_num_policy', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=300)
    opt = parser.parse_args()
    # example
    # python train.py --continue_training --load_epoch_num 0 --save_freq 1 --continue_training_policy  --load_epoch_num_policy 2
    return opt
