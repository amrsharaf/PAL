import numpy as np
import logging
import re
import itertools
from collections import Counter
import tensorflow as tf

# map a label to a string
label2str = {1: "PER", 2: "LOC", 3: "ORG", 4: "MISC", 5: "O"}

# predefine a label_set: PER - 1, LOC - 2, ORG - 3, MISC - 4, O - 5, 0 is for padding
labels_map = {'B-ORG': 3, 'O': 5, 'B-MISC': 4, 'B-PER': 1, 'I-PER': 1, 'B-LOC': 2, 'I-ORG': 3, 'I-MISC': 4, 'I-LOC': 2}


def load_data2labels(input_file):
    print('loading data from: ', input_file)
    seq_set = []
    seq = []
    seq_set_label = []
    seq_label = []
    seq_set_len = []
    with open(input_file, "r", encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == "":
                seq_set.append(" ".join(seq))
                seq_set_label.append(seq_label)
                seq_set_len.append(len(seq_label))
                seq = []
                seq_label = []
            else:
                tok, label = line.split()
                seq.append(tok)
                seq_label.append(labels_map[label])
    return [seq_set, seq_set_label, seq_set_len]


def remove_language_prefix(embedding):
    return embedding[3:]


# TODO pickle the output of this function to make the performance faster
def load_crosslingual_embeddings(input_file, vocab, max_vocab_size=20000, emb_size=40):
    embeddings = list(open(input_file, "r", encoding="utf-8").readlines())
    # Pre-process to remove the language prefix
    embeddings = map(remove_language_prefix, embeddings)
    pre_w2v = {}
    for emb in embeddings:
        parts = emb.strip().split()
        # Make sure embeddings have the correct dimensions
        assert emb_size == (len(parts) - 1)
        w = parts[0]
        vec = []
        for i in range(1, len(parts)):
            vec.append(float(parts[i]))
        pre_w2v[w] = vec
    n_dict = len(vocab)
    vocab_w2v = np.zeros((n_dict, emb_size))
    # vocab_w2v[0]=np.random.uniform(-0.25,0.25,100)
    for w, i in vocab.items():
        if w in pre_w2v:
            vocab_w2v[i] = pre_w2v[w]
        else:
            vocab_w2v[i] = list(np.random.uniform(-0.25, 0.25, emb_size))
    cur_i = len(vocab_w2v)
    if len(vocab_w2v) > max_vocab_size:
        print('Vocabulary size is larger than ', max_vocab_size)
        raise SystemExit
    while cur_i < max_vocab_size:
        cur_i += 1
        padding = [0] * emb_size
        vocab_w2v.append(padding)
    logging.info('Vocabulary {} Embedding size {}'.format(n_dict, emb_size))
    return vocab_w2v


def data2sents(X, Y):
    data = []
    for i in range(len(Y)):
        sent = []
        text = X[i]
        items = text.split()
        for j in range(len(Y[i])):
            sent.append((items[j], str(Y[i][j])))
        data.append(sent)
    return data
