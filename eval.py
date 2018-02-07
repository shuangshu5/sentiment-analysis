#-*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import jieba
import sys
reload(sys)
sys.setdefaultencoding('utf8')

checkpoint_dir = 'model'
allow_soft_placement = True
log_device_placement = False

sentence = u"质量很差"
sentence = jieba.lcut(sentence)
vocabulary = []
f = open('data/vocab.txt','r')
for line in f.readlines():
    vocabulary.append(line.strip())

vec = []
for word in sentence:
    try:
        vec.append(vocabulary.index(word))
    except:
        vec.append(1)

all_vec = []
f = open('data/vec.txt','r')
for line in f.readlines():
    all_vec.append(line.split(' ')[:-1])
max_seq_length = max([len(i) for i in all_vec])
inputs_major = np.zeros(shape=[1,max_seq_length], dtype=np.int32)
for v in range(len(vec)):
    inputs_major[0][v] = vec[v]

def softmax(x):
 return np.exp(x)/np.sum(np.exp(x), axis=0)

checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=allow_soft_placement,
      log_device_placement=log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        scores = graph.get_operation_by_name("output/scores").outputs[0]
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        scores = sess.run(scores, {input_x: inputs_major, dropout_keep_prob: 1.0})
        label = sess.run(predictions, {input_x: inputs_major, dropout_keep_prob: 1.0})
        prob = softmax(scores[0])

        if label[0] == 0:
            label = u'负面情感'
            prob = prob[0]
        else:
            label = u'正向情感'
            prob = prob[1]
        print u'情感倾向: %s   置信度: %f' % (label, prob)


