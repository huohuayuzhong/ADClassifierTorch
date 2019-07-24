import torch
import SimpleITK as sitk
import os
from csv import reader
from torch.utils.data.dataset import Dataset
import numpy as np


# Baseline
class ReadImagesBaseLine(Dataset):
    def __init__(self, data, train, test=False, neg_type='CN', pos_type='AD', cv=1):
        self.__MRI = []
        self.__PET = []
        self.__type = []
        self.__label = []
        self.data = data
        self.train = train
        self.test = test
        self.nt = neg_type
        self.pt = pos_type
        self.cv = cv

        mri_path = os.path.join(self.data, 'ROI_MRI')
        pet_path = os.path.join(self.data, 'ROI_PET')
        label_dict = {self.nt: 0, self.pt: 1}

        # 根据按照病人的ID，事先给每张图片赋予了随机数，同一个病人的不同图片对应的随机数是相同的
        with open('ADNI_all.csv', 'r', encoding='utf-8') as f:
            r = reader(f)
            data = []
            for row in r:
                data.append(row)
            header = data[0]
            print(header[1], header[2], header[10], header[11])
            # 去掉标题
            data = data[1:]

        # 根据随机数的值分组
        if self.train:
            l_bound = 0.0
            u_bound = 0.7
        elif self.test:
            l_bound = 0.8
            u_bound = 1.0
        else:
            l_bound = 0.7
            u_bound = 0.8

        for row in data:
            # type
            if row[10] in label_dict:
                temp_label = label_dict[row[10]]
                if l_bound < float(row[11]) < u_bound:
                    self.__MRI.append(os.path.join(mri_path, row[1]))
                    self.__PET.append(os.path.join(pet_path, row[2]))
                    self.__type.append(row[10])
                    self.__label.append(temp_label)
            else:
                continue

        assert len(self.__MRI) == len(self.__PET) == len(self.__label), \
            'The number of images and labels are not equal'
        self.dataset_size = len(self.__MRI)

    def __getitem__(self, index):
        mri_image = sitk.ReadImage(self.__MRI[index])
        pet_image = sitk.ReadImage(self.__PET[index])
        mri_image = sitk.GetArrayFromImage(mri_image)
        pet_image = sitk.GetArrayFromImage(pet_image)
        subject_type = self.__type[index]
        label = self.__label[index]

        # 数据都是预处理好并且归一化到[0, 1024]的，因此不用做预处理
        mri_image = mri_image.astype(np.float32) / 1024
        pet_image = pet_image.astype(np.float32) / 1024

        return mri_image, pet_image, label

    def __len__(self):
        return len(self.__MRI)


# 5-fold CV
class ReadImagesCrossValidation(Dataset):
    def __init__(self, data, train, test=False, neg_type='CN', pos_type='AD', cv=1):
        self.__MRI = []
        self.__PET = []
        self.__type = []
        self.__label = []
        self.data = data
        self.train = train
        self.test = test
        self.nt = neg_type
        self.pt = pos_type
        self.cv = cv

        mri_path = os.path.join(self.data, 'ROI_MRI')
        pet_path = os.path.join(self.data, 'ROI_PET')
        label_dict = {self.nt: 0, self.pt: 1}

        # 根据按照病人的ID，事先给每张图片赋予了随机数，同一个病人的不同图片对应的随机数是相同的
        with open('ADNI_all.csv', 'r', encoding='utf-8') as f:
            r = reader(f)
            data = []
            for row in r:
                data.append(row)
            header = data[0]
            print(header[1], header[2], header[10], header[11])
            # 去掉标题
            data = data[1:]

        # 根据随机数的值分组
        if self.train:
            l_bound = 0.0
            u_bound = 0.7
        elif self.test:
            l_bound = 0.8
            u_bound = 1.0
        else:
            l_bound = 0.7
            u_bound = 0.8

        # 确定test 和 val的范围
        test_interval = 5 - cv
        val_interval = 2 * test_interval - 1 if test_interval > 0 else 9

        if self.test:
            for row in data:
                # type
                if row[10] in label_dict:
                    temp_label = label_dict[row[10]]
                    if int(5*float(row[11])) % 5 == test_interval:
                        self.__MRI.append(os.path.join(mri_path, row[1]))
                        self.__PET.append(os.path.join(pet_path, row[2]))
                        self.__type.append(row[10])
                        self.__label.append(temp_label)
                else:
                    continue
        elif self.train:
            for row in data:
                # type
                if row[10] in label_dict:
                    temp_label = label_dict[row[10]]
                    if not (int(5*float(row[11])) % 5 == test_interval or
                            int(10*float(row[11])) % 10 == val_interval):
                        self.__MRI.append(os.path.join(mri_path, row[1]))
                        self.__PET.append(os.path.join(pet_path, row[2]))
                        self.__type.append(row[10])
                        self.__label.append(temp_label)
                else:
                    continue
        else:
            for row in data:
                # type
                if row[10] in label_dict:
                    temp_label = label_dict[row[10]]
                    if int(10*float(row[11])) % 10 == val_interval:
                        self.__MRI.append(os.path.join(mri_path, row[1]))
                        self.__PET.append(os.path.join(pet_path, row[2]))
                        self.__type.append(row[10])
                        self.__label.append(temp_label)
                else:
                    continue

        assert len(self.__MRI) == len(self.__PET) == len(self.__label), \
            'The number of images and labels are not equal'
        self.dataset_size = len(self.__MRI)

    def __getitem__(self, index):
        mri_image = sitk.ReadImage(self.__MRI[index])
        pet_image = sitk.ReadImage(self.__PET[index])
        mri_image = sitk.GetArrayFromImage(mri_image)
        pet_image = sitk.GetArrayFromImage(pet_image)
        subject_type = self.__type[index]
        label = self.__label[index]

        # 数据都是预处理好并且归一化到[0, 1024]的，因此不用做预处理
        mri_image = mri_image.astype(np.float32) / 1024
        pet_image = pet_image.astype(np.float32) / 1024

        return mri_image, pet_image, label

    def __len__(self):
        return len(self.__MRI)
