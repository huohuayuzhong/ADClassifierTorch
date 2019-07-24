import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block_3d(in_dim, out_dim, stride=1, padding=1, batch_norm=True):
    '''
    A standard 3d Conv block

    :param in_dim: in_channels
    :param out_dim:  out_channels
    :param stride:  stride
    :param padding:  padding
    :param batch_norm: whether use bn
    :return: model itself
    '''
    if batch_norm:
        conv_block = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=stride, padding=padding),
            # 在tf的版本里BN是加在ReLU后面的，如果实验结果有差异，可以试试看
            # nn.ReLU(),
            nn.BatchNorm3d(out_dim),
            nn.ReLU()
        )
    else:
        conv_block = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=stride, padding=padding),
            nn.ReLU()
        )
    return conv_block


def baseline_conv_layers(init_kernel=16):
    '''
    The network baseline

    :param init_kernel:
    :return: model itself
    '''
    bl_conv = nn.Sequential(
        # Conv1
        conv_block_3d(1, init_kernel),
        nn.MaxPool3d(2, stride=2),
        # Conv2
        conv_block_3d(init_kernel, 2 * init_kernel),
        nn.MaxPool3d(2, stride=2),
        # Conv3
        conv_block_3d(2 * init_kernel, 4 * init_kernel),
        conv_block_3d(4 * init_kernel, 4 * init_kernel),
        nn.MaxPool3d(2, stride=2),
        # Conv4
        conv_block_3d(4 * init_kernel, 8 * init_kernel),
        conv_block_3d(8 * init_kernel, 8 * init_kernel),
        nn.MaxPool3d(2, stride=2),
        # Conv5
        conv_block_3d(8 * init_kernel, 8 * init_kernel),
        conv_block_3d(8 * init_kernel, 8 * init_kernel),
        nn.MaxPool3d(2, stride=2)
    )
    return bl_conv


# for network A, as A have 2 input channels
def baseline_conv_layers_A(init_kernel=16):
    '''
    The network baseline

    :param init_kernel:
    :return: model itself
    '''
    bl_conv = nn.Sequential(
        # Conv1
        conv_block_3d(2, init_kernel),
        nn.MaxPool3d(2, stride=2),
        # Conv2
        conv_block_3d(init_kernel, 2 * init_kernel),
        nn.MaxPool3d(2, stride=2),
        # Conv3
        conv_block_3d(2 * init_kernel, 4 * init_kernel),
        conv_block_3d(4 * init_kernel, 4 * init_kernel),
        nn.MaxPool3d(2, stride=2),
        # Conv4
        conv_block_3d(4 * init_kernel, 8 * init_kernel),
        conv_block_3d(8 * init_kernel, 8 * init_kernel),
        nn.MaxPool3d(2, stride=2),
        # Conv5
        conv_block_3d(8 * init_kernel, 8 * init_kernel),
        conv_block_3d(8 * init_kernel, 8 * init_kernel),
        nn.MaxPool3d(2, stride=2)
    )
    return bl_conv


class NetworkB2(nn.Module):
    def __init__(self, init_kernel, device, n_output=2):
        super(NetworkB2, self).__init__()
        self.init_kernel = init_kernel
        self.device = device
        self.n_output = n_output

        # mri
        self.mri_conv = baseline_conv_layers(init_kernel)
        # pet
        self.pet_conv = baseline_conv_layers(init_kernel)

        # fc layers
        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 1 * 8 * init_kernel * 2, 512),
            nn.Dropout(),
            nn.Linear(512, 10),
            nn.Dropout(),
            nn.Linear(10, n_output),
        )

    def forward(self, inputs):
        mri, pet, label = inputs

        # [B, 48, 96, 96] -> [B, 1, 48, 96, 96]
        mri = mri.unsqueeze(1)
        pet = pet.unsqueeze(1)

        mri_feat = self.mri_conv(mri)
        pet_feat = self.pet_conv(pet)
        mri_feat = mri_feat.view(mri.size(0), -1)
        pet_feat = pet_feat.view(pet.size(0), -1)
        feature_map = torch.cat([mri_feat, pet_feat], 1)
        fc_out = self.fc(feature_map)
        # 在tf的版本里是softmax
        output = F.log_softmax(fc_out)
        return output


class NetworkB1(nn.Module):
    def __init__(self, init_kernel, device, n_output=2):
        super(NetworkB1, self).__init__()
        self.init_kernel = init_kernel
        self.device = device
        self.n_output = n_output

        # share conv kernels
        self.conv = baseline_conv_layers(init_kernel)

        # fc layers
        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 1 * 8 * init_kernel * 2, 512),
            nn.Dropout(),
            nn.Linear(512, 10),
            nn.Dropout(),
            nn.Linear(10, n_output),
        )

    def forward(self, inputs):
        mri, pet, label = inputs

        # [B, 48, 96, 96] -> [B, 1, 48, 96, 96]
        mri = mri.unsqueeze(1)
        pet = pet.unsqueeze(1)

        mri_feat = self.conv(mri)
        pet_feat = self.conv(pet)
        mri_feat = mri_feat.view(mri.size(0), -1)
        pet_feat = pet_feat.view(pet.size(0), -1)
        feature_map = torch.cat([mri_feat, pet_feat], 1)
        fc_out = self.fc(feature_map)
        # 在tf的版本里是softmax
        output = F.log_softmax(fc_out)
        return output


class NetworkA(nn.Module):
    def __init__(self, init_kernel, device, n_output=2):
        super(NetworkA, self).__init__()
        self.init_kernel = init_kernel
        self.device = device
        self.n_output = n_output

        # share conv kernels
        self.conv = baseline_conv_layers_A(init_kernel)

        # fc layers
        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 1 * 8 * init_kernel, 512),
            nn.Dropout(),
            nn.Linear(512, 10),
            nn.Dropout(),
            nn.Linear(10, n_output),
        )

    def forward(self, inputs):
        mri, pet, label = inputs

        # [B, 48, 96, 96] -> [B, 1, 48, 96, 96]
        mri = mri.unsqueeze(1)
        pet = pet.unsqueeze(1)

        img = torch.cat([mri, pet], 1)

        img_feat = self.conv(img)

        img_feat = img_feat.view(mri.size(0), -1)
        fc_out = self.fc(img_feat)
        # 在tf的版本里是softmax
        output = F.log_softmax(fc_out)
        return output
