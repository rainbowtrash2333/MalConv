import os
import tensorflow as tf
import pandas as pd

source = 'D:/TEMP/data/train/train'
label_Path = 'D:/TEMP/data/trainLabels.csv'

file_size = 2000000


def string_to_hexsarray(str):
    return [s for s in str.split() if len(s) == 2 and s != "??"]


def read_file(entry):
    with open(entry, 'r') as f:
        return f.read()


def hexarray_to_bytes(hexarray):
    while len(hexarray) < file_size:
        hexarray.append('100')
    str = " ".join(hexarray)
    return str


def string_to_bytes(str):
    return hexarray_to_bytes(string_to_hexsarray(str))


# 获取符合要求的文件及其标签

def open_files_and_get_label(dir_path, label_path, length=2000, size=2000000, end="bytes"):
    files = []
    labels = []
    i = 0

    df = pd.read_csv(label_path)
    with os.scandir(dir_path) as dir:
        for entry in dir:
            i += 1
            if i < 4000:
                print(11)
                continue

            if entry.name.endswith(end):
                if os.path.getsize(entry) <= size:
                    files.append(entry)
                    file_name = entry.name.split('.')[0]
                    print(file_name)
                    index = df[df['Id'] == file_name].index.tolist()[0]
                    print(df.iat[index, 1])
                    labels.append(df.iat[index, 1])
                    if len(files) == length:
                        break
        return files, labels


# 产生数据

"""
# source_path 数据集目录
# label_path 存储标签的csv文件路径
# length 数据集的数量，默认2000条
# size 文件最大的大小
# end 文件结尾
"""


def produce_data(source_path, label_path, length=2000, size=2000000, end="bytes"):
    data, label = open_files_and_get_label(source_path, label_path, length, size, end)
    data = list(map(read_file, data))
    data = list(map(string_to_bytes, data))
    data = dict(zip(data, label))
    return data


# 产生TFRecord文件

"""
data 数据
filename 产生TFRecord文件的路径
"""


def produce_TFRecord(data, filename):
    print("write tfrecord")
    writer = tf.io.TFRecordWriter(filename)
    for train, label in data.items():
        train = bytes(train, encoding="utf8")
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    'train': tf.train.Feature(bytes_list=tf.train.BytesList(value=[train]))
                }
            )
        )

        # 将序列转为字符串

        writer.write(example.SerializeToString())
    writer.close()


def _parse_function(example):
    features = {'label': tf.FixedLenFeature([], tf.int64),
                'train': tf.FixedLenFeature([], tf.string)}
    parsed_features = tf.parse_single_example(example, features)
    # Perform additional preprocessing on the parsed data.

    label = tf.cast(parsed_features["label"], tf.int32)
    train = tf.cast(parsed_features["train"], tf.string)
    return train, label


def string_to_int(s):
    return [int(i, 16) for i in str(s, encoding='utf-8').split()]


# 例子

if __name__ == "__main__":
    data = produce_data(source, label_Path, length=1000)
    produce_TFRecord(data, 'D:/TEMP/1000_11_test.tfrecord')
