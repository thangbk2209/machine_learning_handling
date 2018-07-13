import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf 
number_of_points = 500
x_point = []
y_point = []
a = np.array([0.22,0.28])
b = 0.78
for i in range(number_of_points):
    x = np.array([np.random.normal(0.0,0.5), np.random.normal(0.0,0.5)])
    # print x.transpose()
    y = a.dot(x.transpose()) + b +np.random.normal(0.0,0.1)
    print y
    x_point.append(x)
    y_point.append([y])
# plt.plot(x_point,y_point, 'o', label='Input Data')
# plt.legend()
# plt.show()
x_point = np.array(x_point)
print x_point
y_point = np.array(y_point)
# print y_point
a = tf.Variable(tf.random_uniform([2], -1.0, 1.0),2)
b = tf.Variable(tf.zeros([1]))
y = tf.multiply(a,x_point) + b
# Danh gia gia tri cost function bang ham mean square error
cost_function = tf.reduce_mean(tf.square(y - y_point))
# toi uu trong so phuong phap gradient descent
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(cost_function)
model = tf.initialize_all_variables()
with tf.Session() as session:
    session.run(model)
    print session.run(y)
    for step in range(0,200):
        session.run(train)
        if (step % 5) == 0:
            print session.run(cost_function)
            plt.plot(x_point,y_point,'o',
                label='step = {}'
                .format(step))
            plt.plot(x_point, session.run(a) *x_point +session.run(b))
            plt.legend()
            plt.show()
            print session.run(a)
            print session.run(b)