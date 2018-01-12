import tensorflow as tf
import numpy as np


gph = tf.Graph()
with gph.as_default():
    pass


with tf.Session(gph) as sess:
    sess.run(tf.global_variables_initializer())
    pass

