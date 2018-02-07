#-*- coding: UTF-8 -*-
import numpy as np
import collections

# Load data
def load_data_and_labels(positive_data_file, negative_data_file, alpha):
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [[s.strip()] for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Sample positive_examples
    positive_indices = np.random.permutation(np.arange(len(positive_examples)))
    positive_examples = np.array(positive_examples)[positive_indices]
    positive_examples = positive_examples[:int(alpha * len(negative_examples))]
    positive_examples = [s[0] for s in positive_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    print 'total_example: %d' % len(y)
    print 'positive_examples: %d' % len(positive_examples), 'negative_examples: %d' % len(negative_examples)
    return [x_text, y]

# create vocab
def create_vocab(x_text, min_counts, vocabFile):
    vocabulary = []
    vocab = ['__PAD__', '__UNK__']
    vocab_file = open(vocabFile, "w")
    for sent in x_text:
        vocabulary.extend(sent.split(' '))
    vocabulary = collections.Counter(vocabulary)
    vocabulary = {k: v for k, v in vocabulary.items() if v > min_counts}
    vocabulary = sorted(vocabulary.items(), key=lambda e: e[1], reverse=True)
    vocabulary = [i[0] for i in vocabulary]
    vocabulary = vocab + vocabulary
    print u'词汇表共有%d个词汇' % len(vocabulary)
    for word in vocabulary:
        vocab_file.write(word + '\n')
    vocab_file.close()
    return vocabulary, len(vocabulary)

# vocab2vec
def vocab2vec(x_text, vocabulary, vecFile):
    word_dicts = {}
    vec = []
    vec_file = open(vecFile, "w")
    for index, word in enumerate(vocabulary):
        word_dicts[word] = index
    for sent in x_text:
        sent_vec = []
        for word in sent.split(' '):
            v = word_dicts.get(word, 1)
            sent_vec.append(v)
            vec_file.write(str(v) + ' ')
        vec.append(sent_vec)
        vec_file.write('\n')
    vec_file.close()
    vec = np.array(vec)
    return vec

# Get batch
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
