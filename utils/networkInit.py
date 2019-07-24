import torch
import numpy as np


def modify_learning_rate(optimizer, init_lr=1e-4, warp_up=False, steps='', epoch=1):
    '''
    Aplly optimizer and learning rate decay to the lr in the training

    :param optimizer: the instance of the optimizer that used in the training
    :param init_lr: the initial learning rate
    :param warp_up: wether use warm up or not
    :param steps: a str that indicates the steps to decay, e.g. '10, 30'
    :param epoch: current epoch
    :return: None (directly change the params in the optimizer)
    '''

    if warp_up:
        if epoch < 10:
            lr = 0.1 * init_lr * pow(1.25, epoch)
            print('Current learning rate: ', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if epoch == 10:
            lr = init_lr
            print('Current learning rate: ', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    if steps:
        step_list = [int(s) for s in steps.split(',')]
        for s in step_list:
            if epoch == s:
                lr = init_lr * pow(0.3, step_list.index(s) + 1)
                print("learning rate:", lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr


def weights_init(m):
    '''
    Intialize weights

    :param m: the input model
    :return: None
    '''
    if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.ConvTranspose3d):
        torch.nn.init.xavier_normal_(m.weight.data, gain=np.sqrt(2.0))
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.1)
    elif isinstance(m, torch.nn.BatchNorm3d) and m.affine:
        torch.nn.init.constant_(m.weight.data, 1.0)
        torch.nn.init.constant_(m.bias.data, 0.0)