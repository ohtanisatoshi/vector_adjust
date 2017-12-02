'''
vec_adjust_test.py

Copyright (c) 2017 Satoshi Otani

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
'''
#! -*- coding:utf-8 -*-
import math
import numpy as np
import tensorflow as tf

class AdjustVector:
    def __init__(self, embedding_size=300, epoch_count=20000, learning_rate=0.5, batch_size=10):
        self._embedding_size = embedding_size
        self._epoch_count = epoch_count
        self._learning_rate = learning_rate
        self._batch_size = batch_size

    def get_adjust_vector(self, target_vector, other_vector_and_cos):
        target_vector = np.array(target_vector)
        other_vectors = np.array([v for v, c in other_vector_and_cos])
        other_vectors_cos = np.array([[c] for v, c in other_vector_and_cos])

        # other_vector = {vector: cos, ...}
        other_vec_size = len(other_vector_and_cos)
        # 正規化
        target_vector = target_vector / np.linalg.norm(target_vector)
        other_vectors = other_vectors / np.linalg.norm(other_vectors, axis=1).reshape(other_vec_size, 1)

        # tensorflow用の入力を準備
        x1 = np.array([target_vector for i in range(len(other_vectors))])
        x2 = other_vectors
        t = other_vectors_cos

        input_1 = tf.placeholder(tf.float32, shape=[None, self._embedding_size])
        input_2 = tf.placeholder(tf.float32, shape=[None, self._embedding_size])
        input_t = tf.placeholder(tf.float32, shape=[None, 1])

        w = tf.Variable(tf.ones([self._embedding_size]))
        b = tf.Variable(tf.zeros([1]))
        y = tf.matmul((input_2), tf.transpose(input_1*w+b))

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        loss = tf.reduce_mean(tf.square((y-input_t)))
        train_step = tf.train.GradientDescentOptimizer(self._learning_rate).minimize(loss)
        for epoch in range(self._epoch_count):
            rand_index = np.arange(0, len(x2), 1, dtype=np.int)
            np.random.shuffle(rand_index)
            for i in range(math.ceil(float(len(x2))/self._batch_size)):
                begin_index = i * self._batch_size
                end_index = min(i * self._batch_size + self._batch_size, len(x2))
                index_for_this_batch = rand_index[begin_index:end_index]
                batch_x1 = x1[index_for_this_batch]
                batch_x2 = x2[index_for_this_batch]
                batch_t = t[index_for_this_batch]
                feed_dict={
                    input_1: batch_x1,
                    input_2: batch_x2,
                    input_t: batch_t
                }

                sess.run(train_step, feed_dict=feed_dict)

            if epoch % 1000 == 0:
                feed_dict={
                    input_1: x1,
                    input_2: x2,
                    input_t: t
                }
                l = sess.run(loss, feed_dict=feed_dict)
                print('{:.8f}'.format(l))

        return_w = sess.run(w)
        return_b = sess.run(b)

        return return_w, return_b


if __name__ == '__main__':
    other_vec_size = 20
    embedding_size = 300

    my_vec = np.random.rand(embedding_size)*40-20
    other_vecs = [np.random.rand(embedding_size)*40-20 for i in range(other_vec_size)]

    my_vec = my_vec / np.linalg.norm(my_vec)
    other_vecs = other_vecs / np.linalg.norm(other_vecs, axis=1).reshape(other_vec_size, 1)

    cos = np.dot(other_vecs, my_vec.T)
    print(cos)
    for i, c in enumerate(cos):
        cos[i] -= np.random.rand(1)*2-1
    print(cos)

    vec_and_cos = [(v, c) for v, c in zip(other_vecs, cos)]

    adjust_vector = AdjustVector(embedding_size=embedding_size, epoch_count=20000)
    w, b = adjust_vector.get_adjust_vector(my_vec, vec_and_cos)
    print('answer: {}'.format(cos))
    print('result: {}'.format(np.dot(other_vecs, (my_vec*w+b).T)))



