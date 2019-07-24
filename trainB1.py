import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import numpy as np
# python files in this project
from utils.networkInit import *
from network.multiModalityNetworks import NetworkB1
from dataset.data import ReadImagesBaseLine


parser = argparse.ArgumentParser(description='AD Classifier')
parser.add_argument('--batchsize', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--load', action='store_true', default=False,
                    help='enables load weights')
parser.add_argument('--load_model', type=str, default=' ', metavar='str',
                    help='the directory of the saved models')
parser.add_argument('--warm_up', action='store_true', default=False,
                    help='enables warm_up')
parser.add_argument('--log_interval', type=int, default=4, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--data', type=str, default='/data/ADNI_ROI', metavar='str',
                    help='folder that contains data (default: test dataset)')
parser.add_argument('--save_dir', type=str, default='/data/st/ADNI/', metavar='str',
                    help='folder that save model')
parser.add_argument('--network', type=str, default='B1', metavar='str',
                    help='the network name')
parser.add_argument('--optim', type=str, default='ADAM', metavar='str',
                    help='the optimizer')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='the learning rate')
parser.add_argument('--step', type=str, default='10, 30',
                    help='lr_policy')
parser.add_argument('--gpu', type=str, default='0', metavar='N',
                    help='gpu numbers')
parser.add_argument('--initial_kernel', type=int, default=16, metavar='N',
                    help='input_channel')
args = parser.parse_args()
print("Args: ", args)

# GPU and CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
print("The number of GPUs:", torch.cuda.device_count())

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
load_args = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# Initial lr and optimizer
print("The learning rate:", args.lr)
print("The optimizer:", args.optim)

# The network used
if args.network == 'A':
    pass
if args.network == 'B1':
    model = NetworkB1(init_kernel=args.initial_kernel, device=device)
    # pass
if args.network == 'B2':
    # model = NetworkB2(init_kernel=args.initial_kernel, device=device)
    pass

if torch.cuda.device_count() > 0:
    print("Using MultiGPUs")
    model = nn.DataParallel(model)

model.to(device)
model.apply(weights_init)

# Save model dir
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)


# Load saved models
if args.load:
    model.load_state_dict(torch.load(args.load_model))

# Optimizer
if args.optim == 'RMS':
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9)
elif args.optim == 'ADAM':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)


def load_data():

    dset_train = ReadImagesBaseLine(args.data, train=True, neg_type='CN', pos_type='AD')

    train_loader = DataLoader(dset_train, batch_size=args.batchsize, shuffle=True, **load_args)

    dset_val = ReadImagesBaseLine(args.data, train=False, neg_type='CN', pos_type='AD')

    val_loader = DataLoader(dset_val, batch_size=args.batchsize, shuffle=False, **load_args)

    print("Training Data : ", len(train_loader.dataset))
    print("validation Data : ", len(val_loader.dataset))
    return train_loader, val_loader


def train(model, device, train_loader, optimizer, epoch):
    model.train().to(device)
    for batch_idx, (mri, pet, label) in enumerate(train_loader):
        mri, pet, label = mri.to(device), pet.to(device), label.to(device)
        optimizer.zero_grad()
        data = (mri, pet, label)
        output = model(data)
        loss = F.nll_loss(output, label, reduction='mean')
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batchsize, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    # save model
    if epoch % 5 == 0 or epoch > args.epochs - 20:
        model.eval().cpu()
        save_filename = 'ADClassifier_B1' + "_epoch_" + str(epoch) + ".model"
        save_path = os.path.join(args.save_dir, save_filename)
        torch.save(model.state_dict(), save_path)
        print("\nDone, trained model saved at", save_path)


def test(model, device, test_loader, epoch):
    model.eval().to(device)
    test_loss = 0
    correct = 0
    # out_list = []
    # pred_list = []
    # label_list = []
    with torch.no_grad():
        for mri, pet, label in test_loader:
            mri, pet, label = mri.to(device), pet.to(device), label.to(device)
            data = (mri, pet, label)
            output = model(data)
            test_loss += F.nll_loss(output, label, reduction='mean').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()

            # out_list.append(output.item())
            # pred_list.append(pred.item())
            # label_list.append(label.item())

    test_loss /= len(test_loader.dataset)
    print('\nTest {}: Average loss: {:.6f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train_loader, val_loader = load_data()
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, val_loader, epoch)
