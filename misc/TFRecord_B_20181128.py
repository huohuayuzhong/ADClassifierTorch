import os
import nipy
import numpy as np
import tensorflow as tf
from csv import reader
from tqdm import tqdm
from random import shuffle

# 对于双模态的数据
# 这里只是对MRI制作了tfrecords

with open('ADNI_all.csv', 'r', encoding='utf-8') as f:
    r = reader(f)
    data = []
    for row in r:
        data.append(row)
    header = data[0]
    print(header[1], header[2], header[10], header[11])
    data = data[1:]

label_dict = {'pMCI': -1, 'AD': 1}
typeStr = ''.join([key for key in label_dict])
print(label_dict, typeStr)

train = []
val = []
test = []
datasets = {}

for row in data:
    if row[10] in label_dict:
        temp_label = label_dict[row[10]]
        if float(row[11]) < 0.7:
            # continue
            train.append([row[1], row[2], temp_label])
        elif float(row[11]) > 0.8:
            test.append([row[1], row[2], temp_label])
        else:
            val.append([row[1], row[2], temp_label])
    else:
        continue

print('Train: ', len(train))
print('Val: ', len(val))
print('Test: ', len(test))

shuffle(train)
shuffle(val)
shuffle(test)

datasets['Train'] = train
datasets['Val'] = val
datasets['Test'] = test

mri_dir = r'D:/ADNI_ROI/ROI_MRI/'
pet_dir = r'D:/ADNI_ROI/ROI_PET/'
# pet_dir = r'D:/ADNI_ROI/ROI_PET_wholebrain/'

for subset in datasets:
    print('Subset: ', subset)
    # writer = tf.python_io.TFRecordWriter(r'E:/ADNI/SegHippo/ROI_96_96_48/' + folder + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(r'D:/ADNI_ROI/tfrecords/BothB_' + subset + typeStr + '.tfrecords')
    for item in tqdm(datasets[subset]):
        mri = nipy.load_image(mri_dir + item[0]).get_data()
        pet = nipy.load_image(pet_dir + item[1]).get_data()
        mri_raw = mri.astype(np.int16).tostring()
        pet_raw = pet.astype(np.int16).tostring()
        # print(img.shape)
        # break
        # label = item[1]
        # 96 96 64的参数
        # example = tf.train.Example(
        #     features = tf.train.Features(
        # feature = {'label': tf.train.Feature(int64_list = tf.train.Int64List(value=[int(file_label[item])])),
        #            'img': tf.train.Feature(bytes_list = tf.train.BytesList(value = [img_raw]))}))
        # 96 96 48，主要改变是把标签从int变为float
        example = tf.train.Example(
            features=tf.train.Features(
                feature={'label': tf.train.Feature(float_list=tf.train.FloatList(value=[float(item[2])])),
                         'mri': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mri_raw])),
                         'pet': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pet_raw]))
                         # 'neg-label': tf.train.Feature(float_list=tf.train.FloatList(value=[2 - float(file_label[item])]))
                         }))
        serialized = example.SerializeToString()
        writer.write(serialized)

    writer.close()