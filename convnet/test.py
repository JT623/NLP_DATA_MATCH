import os
import sys
import numpy

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from convnet.graph import Graph
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from utils.load_data import load_char_data

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

p, h, y = load_char_data('input/data/test.csv', data_size=None)

model = Graph()
saver = tf.train.Saver()

with tf.Session()as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, '../output/convnet/convnet_29.ckpt')
    loss, acc, label= sess.run([model.loss, model.acc,model.label],
                         feed_dict={model.p: p,
                                    model.h: h,
                                    model.y:y,
                                    model.keep_prob: 1})
    print('loss: ', loss, ' acc:', acc,'label:',label)


    # label = sess.run(y,feed_dict={model.p:p,model.h:h})
    # print(label)
