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
from network.multiModalityNetworks import NetworkB2
from dataset.data import ReadImagesCrossValidation
from csv import writer
import torchnet
from torchnet.meter.aucmeter import AUCMeter

# 保存更多模型，直接看test



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
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--data', type=str, default='/data/ADNI_ROI', metavar='str',
                    help='folder that contains data (default: test dataset)')
parser.add_argument('--save_dir', type=str, default='/data/st/ADNI/', metavar='str',
                    help='folder that save model')
parser.add_argument('--network', type=str, default='B2', metavar='str',
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
parser.add_argument('--cv', type=int, default=1, metavar='N',
                    help='fold of the cross validation')
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
    pass
if args.network == 'B2':
    model = NetworkB2(init_kernel=args.initial_kernel, device=device)

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

    dset_train = ReadImagesCrossValidation(args.data, train=True, neg_type='CN', pos_type='AD', cv=args.cv)

    train_loader = DataLoader(dset_train, batch_size=args.batchsize, shuffle=True, **load_args)

    dset_val = ReadImagesCrossValidation(args.data, train=False, test=True, neg_type='CN', pos_type='AD', cv=args.cv)

    val_loader = DataLoader(dset_val, batch_size=args.batchsize, shuffle=False, **load_args)

    # dset_test = ReadImagesCrossValidation(args.data, train=False, test=True, neg_type='CN', pos_type='AD', cv=args.cv)
    #
    # test_loader = DataLoader(dset_test, batch_size=args.batchsize, shuffle=False, **load_args)

    print("Training Data : ", len(train_loader.dataset))
    print("validation Data : ", len(val_loader.dataset))
    # print('Test Data: ', len(test_loader.dataset))
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
    model.eval().cpu()
    save_filename = 'ADClassifier_B2_fold_' + str(args.cv) + "_epoch_" + str(epoch) + ".model"
    save_path = os.path.join(args.save_dir, save_filename)
    torch.save(model.state_dict(), save_path)
    print("\nDone, trained model saved at", save_path)


def test(model, device, test_loader, epoch):
    model.eval().to(device)
    test_loss = 0
    correct = 0
    auc_meter = AUCMeter()
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    with torch.no_grad():
        for mri, pet, label in test_loader:
            mri, pet, label = mri.to(device), pet.to(device), label.to(device)
            data = (mri, pet, label)
            output = model(data)
            test_loss += F.nll_loss(output, label, reduction='mean').item()
            pred = output.max(1, keepdim=True)[1]
            pred = pred.view_as(label)
            correct += pred.eq(label).sum().item()
            auc_meter.add(pred, label)

            TP += ((pred == 1) & (label == 1)).cpu().sum().item()
            TN += ((pred == 0) & (label == 0)).cpu().sum().item()
            FP += ((pred == 1) & (label == 0)).cpu().sum().item()
            FN += ((pred == 0) & (label == 1)).cpu().sum().item()


            # out_list.append(output.item())
            # pred_list.append(pred.item())
            # label_list.append(label.item())

    auc, _, _ = auc_meter.value()
    print('TP, TN, FP, FN', TP, TN, FP, FN)
    # test_loss /= len(test_loader.dataset)
    print('Accuracy: {}/{} ({:.4f}%)\tSEN: {:.4f}%\tSPE: {:.4f}%\tAUC: {:.6f}\n'.
          format(correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset),
                 100. * TP / (TP+FN), 100. * TN / (TN+FP), auc))


for epoch in range(1, args.epochs + 1):
    train_loader, val_loader = load_data()
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, val_loader, epoch)
