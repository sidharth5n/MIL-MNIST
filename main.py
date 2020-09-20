from __future__ import print_function

import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from dataloader import MnistBags
from models import MIL
from train_test_script import train, test

# Training settings
parser = argparse.ArgumentParser(description = 'MIL MNIST-Bags')
parser.add_argument('--epochs', type = int, default = 10, help = 'number of epochs to train (default: 10)')
parser.add_argument('--lr', type = float, default = 0.0005, help = 'learning rate (default: 0.0005)')
parser.add_argument('--reg', type = float, default = 10e-5, help = 'weight decay')
parser.add_argument('--target', type = int, default = 9, help = 'Bags have a positive labels if they contain at least one of target')
parser.add_argument('--mean_bag_length', type = int, default = 10, help = 'average bag length')
parser.add_argument('--num_bags_train', type = int, default = 500, help = 'number of bags in training set')
parser.add_argument('--num_bags_test', type = int, default = 50, help = 'number of bags in test set')
parser.add_argument('--seed', type = int, default = 1, help = 'random seed (default: 1)')
parser.add_argument('--model', type = str, default = 'attention',
                    help = 'type of aggregation : attention, gated_attention, noisy and, noisy or, ISR, generalized mean, LSE')

args = parser.parse_args()

# Check whether GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Loading Train and Test Set')

loader_kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

test_loader = DataLoader(MnistBags(target = args.target,
                                   train = False,
                                   mean_bag_length = args.mean_bag_length,
                                   total_bags = args.num_bags_test,
                                   seed = args.seed),
                         batch_size = 1,
                         shuffle = False,
                         **loader_kwargs)

 train_loader = DataLoader(MnistBags(target = args.target,
                                     train = True,
                                     mean_bag_length = args.mean_bag_length,
                                     total_bags = args.num_bags_train,
                                     seed = args.seed),
                           batch_size = 1,
                           shuffle = True,
                           **loader_kwargs)

model = MIL(args.model)

loss = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = args.lr, betas = (0.9, 0.999), weight_decay = args.reg)

print('Training \n--------')
train(model, train_loader, loss, optimizer, device, args.epochs)
print('Testing \n-------')
test(model, test_loader, loss, device)
