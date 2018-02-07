#-*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import os
import data_helpers
from cnn_net import TextCNN

alpha = 2.0
min_counts = 5
num_checkpoints = 3
embedding_dim = 128
num_filters = 256
batch_size = 512
num_epochs = 100
evaluate_every = 100
checkpoint_every = 100
dropout_keep_prob = 0.8
l2_reg_lambda = 0.01
dev_sample_percentage = 0.01
filter_sizes = "3,4,5"
allow_soft_placement = True
log_device_placement = False
out_dir = 'model'
vecFile = 'data/vec.txt'
vocabFile = 'data/vocab.txt'
positive_data_file = 'data/postive.txt'
negative_data_file = 'data/negtive.txt'

# Load data
x_text, y = data_helpers.load_data_and_labels(positive_data_file, negative_data_file, alpha)
vocabulary, vocabulary_size = data_helpers.create_vocab(x_text, min_counts, vocabFile)
vec_list = data_helpers.vocab2vec(x_text, vocabulary, vecFile)

# Padding
x = []
max_seq_length = max([len(i) for i in vec_list])
for sentence in vec_list:
    inputs_major = np.zeros(shape=[max_seq_length], dtype=np.int32)
    for i in range(len(sentence)):
        inputs_major[i] = sentence[i]
    x.append(inputs_major)
x = np.array(x)

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print 'x_train: %d,x_dev: %d' % (len(x_train), len(x_dev))

del x, y, x_shuffled, y_shuffled

# Training
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=allow_soft_placement,
      log_device_placement=log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=vocabulary_size,
            embedding_size=embedding_dim,
            filter_sizes=list(map(int, filter_sizes.split(","))),
            num_filters=num_filters,
            l2_reg_lambda=l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        checkpoint_dir = os.path.abspath(out_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

        sess.run(tf.global_variables_initializer())


        def train_step(x_batch, y_batch):
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: dropout_keep_prob
            }
            _, step, loss, accuracy = sess.run(
                [train_op, global_step, cnn.loss, cnn.accuracy],
                feed_dict)
            print("step {}, train_loss {:g}, train_acc {:g}".format(step, loss, accuracy))

        def dev_step(x_batch, y_batch):
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, loss, accuracy = sess.run(
                [global_step, cnn.loss, cnn.accuracy],
                feed_dict)
            print("step {}, dev_loss {:g}, dev_acc {:g}".format(step, loss, accuracy))

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), batch_size, num_epochs)
        # Training loop. For each batch
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev)
            if current_step % checkpoint_every == 0:
                path = saver.save(sess, checkpoint_dir, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))


