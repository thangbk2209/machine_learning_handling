# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2
import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# df = read_csv('/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thangbk2209/GGClusterTraceDataAnalysis/data/Fuzzy_data_sampling_617685_metric_10min_datetime_origin.csv', header=None, index_col=False, names=colnames, usecols=[0,1], engine='python')

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)
min = 0
max = 0
mean = 0

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat
def scaling_data(X):
    min = np.amin(X)
    max = np.amax(X)
    mean = np.mean(X)
    scale = (X-min)/(max-min)
    return scale, min, max
def get_iris_data():
    """ Read the iris data set and split them into training and test sets """
    colnames = ['time_stamp','taskIndex','machineId','meanCPUUsage','CMU','AssignMem','unmapped_cache_usage','page_cache_usage','max_mem_usage','mean_diskIO_time','mean_local_disk_space','max_cpu_usage','max_disk_io_time','cpi', 'mai','sampling_portion','agg_type','sampled_cpu_usage']
    df = read_csv('./data/google_trace_timeseries/data_resource_usage_10Minutes_6176858948.csv', header=None, index_col=False, names=colnames, engine='python')
    cpu = df['meanCPUUsage'].values.reshape(-1,1)
    mem = df['CMU'].values.reshape(-1,1)

    mem_nomal, minMem, maxMem = scaling_data(mem)
    cpu_nomal, minCPU, maxCPU = scaling_data (cpu)
    # mem_nomal = scaler.fit_transform(mem)
    # cpu_nomal = scaler.fit_transform(cpu)
    train_size = int(len(cpu) * 0.8)
    trainX = []
    trainY = []
    testX = []
    testY = []
    for i in range(train_size):
        trainXi = []
        trainXi.append(cpu_nomal[i])
        trainXi.append(mem_nomal[i])
        trainX.append(trainXi)
        trainY.append(cpu_nomal[i+1])
    for i in range(train_size,len(cpu)-1):
        testXi = []
        testXi.append(cpu_nomal[i])
        testXi.append(mem_nomal[i])
        testX.append(testXi)
        testY.append(cpu[i+1])
    return np.array(trainX), np.array(trainY), np.array(testX), np.array(testY), minCPU, maxCPU

def main():
    train_X,  train_y, test_X, test_y, minCPU,maxCPU = get_iris_data()
    train_X = train_X.reshape(train_X.shape[0],train_X.shape[1])
    test_X = test_X.reshape(test_X.shape[0],test_X.shape[1] )
    learning_rate = 0.001
    training_epochs = 200
    batch_size = 100
    
    display_step = 1
    # Layer's sizes
    n_input = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    n_hidden_1 = 3           # Number of hidden nodes
    n_output = train_y.shape[1]
    print minCPU
    print maxCPU
    print train_X[0]
    print test_X[0]
    print train_y[0]
    print len(train_y)   # Number of outcomes (3 iris flowers)
    print n_input
    print n_output
    # Symbols
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_output])
    h1 = tf.Variable(tf.random_normal([n_input, n_hidden_1], dtype=tf.float32))
    h2 = tf.Variable(tf.random_normal([n_hidden_1, n_output]))
    bias_layer_1 = tf.Variable(tf.random_normal([n_hidden_1]))

    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,h1),bias_layer_1))

    bias_output = tf.Variable(tf.random_normal([n_output]))
    output_layer = tf.nn.sigmoid(tf.matmul(layer_1,h2),bias_output)

    # Backward propagation
    cost    = tf.reduce_mean(tf.add(tf.square(y-output_layer),)
    # optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # Run SGD
    avg_set = []
    epoch_set=[]
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print sess.run(h1)
        print sess.run(h2)
        print sess.run(bias_layer_1)
        # print sess.run(bias_layer_2)
        for epoch in range(training_epochs):
            # Train with each example
            total_batch = int(len(train_X)/batch_size)
            for i in range(total_batch):
                # sess.run(updates)
                avg_cost = 0
                for i in range(total_batch):
                    batch_xs, batch_ys = train_X[i*batch_size:(i+1)*batch_size], train_y[i*batch_size:(i+1)*batch_size]
                    sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
                    avg_cost += sess.run(cost,feed_dict={x: batch_xs,y: batch_ys})/total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print "Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(avg_cost)
                avg_set.append(avg_cost)
                epoch_set.append(epoch+1)
                print "Training phase finished"
        # plt.plot(epoch_set,avg_set, 'o', label='MLP Training phase')
        # plt.ylabel('cost')
        # plt.xlabel('epoch')
        # plt.legend()
        # plt.show()
        print 'h1 ',sess.run(h1)
        # tf.cast(h1, tf.float64)
        print 'h2 ',sess.run(h2)
        print 'bias layer 1 ', sess.run(bias_layer_1)
        # predictions = []
        # for i in range(len(test_X)):
        layer_1_predictioni = tf.nn.sigmoid(tf.add(tf.matmul(tf.cast(test_X,'float'),h1),bias_layer_1))
        predictions = tf.nn.sigmoid(tf.matmul(layer_1_predictioni,h2))
        # predictions = scaler.inverse_transform(predictions)
        predictions = predictions * (maxCPU - minCPU) + minCPU
        error = tf.reduce_sum(tf.abs(tf.subtract(predictions,test_y)))/len(test_y)
        print 'predictions', sess.run(predictions)
        print 'error', sess.run(error)
        # correct_prediction = tf.equal(tf.argmax(output_layer, 1),tf.argmax(y, 1))
    # # evaluating its accuracy
    #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #     print "Model Accuracy:", accuracy.eval({x: mnist.test.images,y: mnist.test.labels})
            # train_accuracy = np.mean(np.square(train_y, axis=1) ==
            #                          sess.run(predict, feed_dict={X: train_X, y: train_y}))
            # test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
            #                          sess.run(predict, feed_dict={X: test_X, y: test_y}))


        sess.close()

if __name__ == '__main__':
    main()