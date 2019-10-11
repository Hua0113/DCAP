

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
with open(r"C:\pypro\brca_go.csv", 'r') as f:
    data = pd.read_csv(f)

print(data.shape)
tcga_input=np.transpose(data)
print(tcga_input.shape)

learning_rate = 0.01
training_epochs = 10
batch_size = 50
display_step = 1
examples_to_show = 10

dropout=0.1
n_input = 60779
scale = 0.0001
# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

# 用字典的方式存储各隐藏层的参数
n_hidden_1 = 500 # 第一编码层神经元个数
n_hidden_2 = 200 # 第二编码层神经元个数
# 权重和偏置的变化在编码层和解码层顺序是相逆的
# 权重参数矩阵维度是每层的 输入*输出，偏置参数维度取决于输出层的单元数
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}

# 每一层结构都是 xW + b
# 构建编码器
def encoder(x):
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# 构建解码器
def decoder(x):
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# 构建模型
#encoder_op = encoder(X)
#decoder_op = decoder(encoder_op)

##################################################################

fc_1 = tf.layers.dense(inputs=X, units=n_hidden_1,
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=scale))
fc_1_out = tf.nn.tanh(fc_1)
fc_1_dropout = tf.layers.dropout(inputs=fc_1_out, rate=dropout)

fc_2 = tf.layers.dense(inputs = fc_1_dropout, units = n_hidden_2, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=scale))
fc_2_out = tf.nn.tanh(fc_2)
encoder_op = tf.layers.dropout(inputs=fc_2_out, rate=dropout)

fc_3 = tf.layers.dense(inputs = encoder_op, units = n_hidden_1, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=scale))
fc_3_out = tf.nn.tanh(fc_3)
fc_3_dropout = tf.layers.dropout(inputs=fc_3_out, rate=dropout)

decoder_op = tf.layers.dense(inputs=fc_3_dropout, units=n_input)
##################################################################

# 预测
y_pred = decoder_op
y_true = X

# 定义代价函数和优化器
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))#+lossL #最小二乘法
l2_loss = tf.losses.get_regularization_loss()
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost+l2_loss)

with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    # 首先计算总批数，保证每次循环训练集中的每个样本都参与训练，不同于批量训练
    total_batch = int(len(tcga_input)/batch_size) #总批数
    for epoch in range(training_epochs):
        for i in range(total_batch):
            # tch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0
            batch_xs = tcga_input[((i)*batch_size):((i+1)*batch_size)]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
        if epoch == training_epochs - 1:
                fea_output = sess.run([encoder_op], feed_dict={X: tcga_input})
                # print(fea_output)
                print(np.array(fea_output).shape)
                np.savetxt(r'C:\pypro\fea.csv', np.array(fea_output[0]), delimiter=',')
    print("Optimization Finished!")

    # encode_decode = sess.run(
    #     y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # f, a = plt.subplots(2, 10, figsize=(10, 2))  #return fig，axes
    # for i in range(examples_to_show):  
    #     a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))  
    #     a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))  
    # plt.show() 