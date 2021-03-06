import logging

import numpy as np

# map a label to a string
#label2str = {1: "PER", 2: "LOC", 3: "ORG", 4: "MISC", 5: "O"}
label2str = {0: "PER", 1: "LOC", 2: "ORG", 3: "MISC", 4: "O"}

# predefine a label_set: PER - 1, LOC - 2, ORG - 3, MISC - 4, O - 5, 0 is for padding
#labels_map = {'B-ORG': 3, 'O': 5, 'B-MISC': 4, 'B-PER': 1, 'I-PER': 1, 'B-LOC': 2, 'I-ORG': 3, 'I-MISC': 4, 'I-LOC': 2}
labels_map = {'B-ORG': 2, 'O': 4, 'B-MISC': 3, 'B-PER': 0, 'I-PER': 0, 'B-LOC': 1, 'I-ORG': 2, 'I-MISC': 3, 'I-LOC': 1}


# TODO we can make this faster, or run it once and pickle the data
def load_data2labels(input_file, max_len):
    logging.debug('loading data from: {}'.format(input_file))
    seq_set = []
    seq = []
    seq_set_label = []
    seq_label = []
    seq_set_len = []
    with open(input_file, "r", encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == "":
                # only store sequences with lenght <= max_len!
                if len(seq_label) <= max_len:
                    seq_set.append(" ".join(seq))
                    # Store labels as np arrays
                    seq_set_label.append(np.array(seq_label))
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
# TODO optimize the runtime pefromance of this function
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
    if n_dict > max_vocab_size:
        logging.debug('Vocabulary size is larger than {}'.format(max_vocab_size))
        raise SystemExit
    vocab_w2v = np.zeros((max_vocab_size, emb_size))
    for w, i in vocab.items():
        if w in pre_w2v:
            vocab_w2v[i] = pre_w2v[w]
        else:
            vocab_w2v[i] = list(np.random.uniform(-0.25, 0.25, emb_size))
    logging.debug('Vocabulary {} Embedding size {}'.format(n_dict, emb_size))
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
