import tensorflow as tf
import numpy as np

file_size = 2000000
padding = 'VALID'


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


lr = tf.Variable(0.001, dtype=tf.float32)

#  input.shape = [batch,2000000]
input_data = tf.placeholder(tf.int32, shape=[None, file_size], name="input")
y = tf.placeholder(tf.float32, shape=[None, 9], name="y")
# training = tf.placeholder("bool", name="training")

input_data1 = tf.reshape(input_data, [-1, file_size])

# 8d-dembedding
embedding = tf.Variable(tf.random_normal([256, 8]), name="embedding")
x = tf.nn.embedding_lookup(embedding, input_data1)

# slice.shape=[branch,2000000,4]
sliceA = tf.slice(x, [0, 0, 0], [-1, -1, 4])
sliceB = tf.slice(x, [0, 0, 4], [-1, -1, 4])

# 初始化第一个卷积层的权值和偏置
W_convl = weight_variable([500, 4, 128])
b_conv1 = bias_variable([128])

V_convl = weight_variable([500, 4, 128])
c_conv1 = bias_variable([128])

# 卷积A
A = conv1d(sliceA, W_convl) + b_conv1
B = conv1d(sliceA, V_convl) + c_conv1

# G0.shape=[branch*4000*128]
G0 = tf.nn.relu(A * tf.nn.sigmoid(B))

P = tf.layers.max_pooling1d(G0, pool_size=4000, strides=1, padding=padding, data_format='channels_last')

W_conv2 = weight_variable([100, 128, 256])
b_conv2 = bias_variable([256])

# 降维,P.shape=[branch*128]
P = tf.reshape(P, [-1, 128])

# FC1
W_fc1 = weight_variable([128, 128])
b_fc1 = bias_variable([128])

h_fc1 = tf.nn.relu(tf.matmul(P, W_fc1))

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# FC2
W_fc2 = weight_variable([128, 9])
b_fc2 = bias_variable([9])

prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y * tf.log(prediction))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def _parse_function(example):
    features = {'label': tf.FixedLenFeature([], tf.int64),
                'train': tf.FixedLenFeature([], tf.string)}
    parsed_features = tf.parse_single_example(example, features)
    # Perform additional preprocessing on the parsed data.
    label = tf.cast(parsed_features["label"], tf.int32)
    train = tf.cast(parsed_features["train"], tf.string)
    return train, label


dataset = tf.data.TFRecordDataset('/input/3000_train.tfrecord')
dataset = dataset.map(_parse_function)
dataset = dataset.repeat()
dataset = dataset.batch(75)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

test = tf.data.TFRecordDataset('/input/1000_test.tfrecord')
test = test.map(_parse_function)
test = test.repeat()
test = test.batch(75)
test_iterator = test.make_initializable_iterator()
test_next_element = iterator.get_next()


def string_to_int(s):
    return [int(i, 16) for i in str(s, encoding='utf-8').split()]


with tf.Session() as sess:
    sess.run(init)
    sess.run(iterator.initializer)
    for epoch in range(21):
        sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))
        for batch in range(40):
            value = sess.run(next_element)
            batch_xs, batch_ys = sess.run([next_element[0], next_element[1]])
            l = []
            for i, j in enumerate(batch_xs):
                l.append(string_to_int(j))
            out = np.asanyarray(l)
            batch_ys = batch_ys - 1
            batch_ys = np.eye(9)[batch_ys.reshape(-1)]
            sess.run(train_step, feed_dict={input_data: out, y: batch_ys, keep_prob: 0.7})

        value = sess.run(test_next_element)
        batch_xs, batch_ys = sess.run([test_next_element[0], test_next_element[1]])
        l = []
        for i, j in enumerate(batch_xs):
            l.append(string_to_int(j))
        out = np.asanyarray(l)
        batch_ys = batch_ys - 1
        batch_ys = np.eye(9)[batch_ys.reshape(-1)]
        test_acc = sess.run(accuracy, feed_dict={input_data: out, y: batch_ys, keep_prob: 1.0})
        print("Iter" + str(epoch) + "Testing Accuracy " + str(test_acc))
