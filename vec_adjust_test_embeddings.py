import numpy as np
import tensorflow as tf

embedding_size = 300
a = np.random.rand(embedding_size)
b = np.random.rand(embedding_size)

a = a / np.linalg.norm(a)
b = b / np.linalg.norm(b)

cos = np.dot(a, b)
print(cos)
cos -= 0.1
print(cos)

x1 = a.reshape([1, embedding_size])
x2 = b.reshape([1, embedding_size])
t = np.array([cos]).reshape([1, 1])

input_1 = tf.placeholder(tf.float64, shape=[None, embedding_size])
input_2 = tf.placeholder(tf.float64, shape=[None, embedding_size])
input_t = tf.placeholder(tf.float64, shape=[None, 1])

w = tf.Variable(input_1)
y = tf.matmul(w, tf.transpose(input_2))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(tf.square((y-t)))
for epoch in range(1000):
    sess.run(train_step, feed_dict={
        input_1: x1,
        input_2: x2,
        input_t: t
    })

answer = y.eval(session=sess, feed_dict={
    input_1: x1,
    input_2: x2,
    input_t: t
})
print(answer)
print(cos)
np_w = sess.run(w)
print(np.dot((a+np_w), b))


