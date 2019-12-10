import tensorflow as tf
import mytool
import numpy as np
import multiprocessing as mp
import os

tf.get_logger().setLevel('ERROR')
np.seterr(divide='ignore', invalid='ignore')

file_size = 2000000
padding = 'VALID'

# train_path = 'D:/TEMP/data/3000_train.tfrecord'
# test_path = 'D:/TEMP/data/1000_test.tfrecord'
# summary_path = './logs/'
# n_epoch = 2
# batch_size = 5
# n_batch = 1
# model_path = './data/MyModel'

n_epoch = 20
batch_size = 100
n_batch = 3000 // batch_size

train_path = '/input/3000_train.tfrecord'
test_path = '/input/1000_test.tfrecord'
summary_path = '/data/summary'
model_path = '/data/MyModel'

class_num = 9
cm_array = np.zeros((n_epoch, class_num, class_num), dtype=int)


# 初始化权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)  # 生成一个正态分布
    return tf.Variable(initial)


# 初始化偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv1d(x, kernel):
    # [batch, in_width, in_channels]
    return tf.nn.conv1d(x, kernel, stride=500, padding=padding)


def max_pool(x):
    return tf.layers.max_pooling1d(x, pool_size=4000, strides=1, padding=padding, data_format='channels_last')


with tf.name_scope('inputs'):
    #  input.shape = [batch,2000000]
    input_data = tf.placeholder(tf.int32, shape=[None, file_size], name="x_input")
    y = tf.placeholder(tf.float32, shape=[None, 9], name="y_input")
    # training = tf.placeholder("bool", name="training")
    lr = tf.Variable(1e-4, dtype=tf.float32, name='lr')
    input_data1 = tf.reshape(input_data, [-1, file_size])

# 8d-dembedding
with tf.name_scope('8-dimensional_embedding_layer'):
    embedding = tf.Variable(tf.random_normal([256, 8]), name="E")
    x = tf.nn.embedding_lookup(embedding, input_data1)

# slice.shape=[branch,2000000,4]
with tf.name_scope('1D_Convolution_layer'):
    with tf.name_scope('slice'):
        sliceA = tf.slice(x, [0, 0, 0], [-1, -1, 4], name='sliceA')
        sliceB = tf.slice(x, [0, 0, 4], [-1, -1, 4], name='sliceB')

    # 初始化第一个卷积层的权值和偏置
    with tf.name_scope('weights'):
        W_convl = weight_variable([500, 4, 128])
        tf.summary.histogram('weights1', W_convl)
        V_convl = weight_variable([500, 4, 128])
        tf.summary.histogram('weights2', V_convl)

    with tf.name_scope('biases'):
        b_conv1 = bias_variable([128])
        tf.summary.histogram('biases1', b_conv1)
        c_conv1 = bias_variable([128])
        tf.summary.histogram('biases2', c_conv1)

    # 卷积A
    with tf.name_scope('Convolution'):
        A = conv1d(sliceA, W_convl) + b_conv1
        B = conv1d(sliceA, V_convl) + c_conv1

# G0.shape=[branch*4000*128]
with tf.name_scope('Gating'):
    G0 = tf.nn.relu(A * tf.nn.sigmoid(B))

with tf.name_scope('max_pooling'):
    P = tf.layers.max_pooling1d(G0, pool_size=4000, strides=1, padding=padding, data_format='channels_last')

with tf.name_scope('128-dim_FC_layer'):
    # 降维,P.shape=[branch*128]
    P = tf.reshape(P, [-1, 128])
    # FC1
    W_fc1 = weight_variable([128, 128])
    b_fc1 = bias_variable([128])

    h_fc1 = tf.nn.relu(tf.matmul(P, W_fc1))

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('9-dim_FC_layer'):
    # FC2
    W_fc2 = weight_variable([128, 9])
    b_fc2 = bias_variable([9])

with tf.name_scope('train'):
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    cross_entropy = -tf.reduce_sum(y * tf.log(prediction))
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    tf.summary.scalar("lr", lr)

with tf.name_scope('result'):
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    tf.summary.scalar("Testing Accuracy", accuracy)

    predicted = tf.one_hot(tf.argmax(prediction, 1), 10)
    actual = tf.one_hot(tf.argmax(y, 1), 10)

    TP = tf.count_nonzero(predicted * actual)
    TN = tf.count_nonzero((predicted - 1) * (actual - 1))
    FP = tf.count_nonzero(predicted * (actual - 1))
    FN = tf.count_nonzero((predicted - 1) * actual)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    tf.summary.scalar("Testing precision", precision)
    tf.summary.scalar("Testing recall", recall)
    tf.summary.scalar("Testing f1", f1)


def _parse_function(example):
    features = {'label': tf.FixedLenFeature([], tf.int64),
                'train': tf.FixedLenFeature([], tf.string)}
    parsed_features = tf.parse_single_example(example, features)
    # Perform additional preprocessing on the parsed data.
    label = tf.cast(parsed_features["label"], tf.int32)
    train = tf.cast(parsed_features["train"], tf.string)
    return train, label


with tf.name_scope('data_process'):
    dataset = tf.data.TFRecordDataset(train_path)
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    test = tf.data.TFRecordDataset(test_path)
    test = test.map(_parse_function)
    test = test.repeat()
    test = test.batch(batch_size)
    test_iterator = test.make_initializable_iterator()
    test_next_element = iterator.get_next()


def string_to_int(s):
    return [int(i, 16) for i in str(s, encoding='utf-8').split()]


def _get_data(session, element):
    session.run(element)
    batch_xs, batch_ys = session.run([element[0], element[1]])
    pool = mp.Pool()
    result = [pool.apply_async(string_to_int, args=(j,)) for i, j in enumerate(batch_xs)]
    batch_ys = batch_ys - 1
    batch_ys = np.eye(9)[batch_ys.reshape(-1)]
    pool.close()
    pool.join()

    out = [i.get() for i in result]
    out = np.asanyarray(out)
    return out, batch_ys


from sklearn.metrics import *

if __name__ == '__main__':
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(summary_path, sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        for epoch in range(n_epoch):
            print("训练了第", epoch, "轮")
            sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))
            for batch in range(n_batch):
                if batch % 10 == 0:
                    print("训练了第", epoch, "组")
                _x, _y = _get_data(sess, next_element)
                sess.run(train_step, feed_dict={input_data: _x, y: _y, keep_prob: 0.7})

            saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)
            saver.save(sess, model_path)

            _x, _y = _get_data(sess, test_next_element)
            result = sess.run(merged, feed_dict={input_data: _x, y: _y, keep_prob: 1.0})
            writer.add_summary(result, epoch)
            pre = sess.run(prediction, feed_dict={input_data: _x, y: _y, keep_prob: 1.0})

            with sess.as_default():
                predict_labels = tf.argmax(pre, 1).eval()
                correct_labels = tf.argmax(_y, 1).eval()

            cm = confusion_matrix(correct_labels, predict_labels, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])

            labels = ['Ramnit', 'Lollipop', 'Kelihos_ver3', 'Vundo', 'Simda',
                      'Tracur', 'Kelihos_ver1', 'Obfuscator.ACY', 'Gatak']
            confusion_matrix_summary = mytool.confusion_matrix_heatmap(cm, labels)
            with tf.name_scope('confusion_matrix'):
                writer.add_summary(confusion_matrix_summary, epoch)
        with tf.name_scope('result'):
            summary_list = mytool.drawing(cm_array, labels=labels)
            for s in summary_list:
                writer.add_summary(s)
        writer.close()

    # 关机
    os.system('./shutdown.sh')
