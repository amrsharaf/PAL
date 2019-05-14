from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
import pycrfsuite
import os
import helpers
import random
import math
import logging
import tensorflow as tf
import numpy as np
import os


# TODO do we need POS features?
# POS was commented out in previous versions of the code
def word2features(sent, i):
    assert len(sent[i]) == 2
    assert type(sent[i][0] == str)
    word = sent[i][0]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        ]
    if i > 0:
        word1 = sent[i - 1][0]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            ])
    else:
        features.append('BOS')
    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            ])
    else:
        features.append('EOS')
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, label in sent]


# TODO seems like this function is not used!
def sent2tokens(sent):
    return [token for token, label in sent]


# Compute f-score
def compute_fscore(Y_pred, Y_true):
    pre = 0
    pre_tot = 0
    rec = 0
    rec_tot = 0
    corr = 0
    total = 0
    number_of_examples = len(Y_true)
    for i in range(number_of_examples):
        sentence_length = len(Y_true[i])
        for j in range(sentence_length):
            total += 1
            if Y_pred[i][j] == Y_true[i][j]:
                corr += 1
            if helpers.label2str[int(Y_pred[i][j])] != 'O':  # not 'O'
                pre_tot += 1
                if Y_pred[i][j] == Y_true[i][j]:
                    pre += 1
            if helpers.label2str[int(Y_true[i][j])] != 'O':
                rec_tot += 1
                if Y_pred[i][j] == Y_true[i][j]:
                    rec += 1
    res = corr * 1. / total
    logging.info('Accuracy (token level) {}'.format(res))
    if pre_tot == 0:
        pre = 0
    else:
        pre = 1. * pre / pre_tot
    rec = 1. * rec / rec_tot
    logging.info('Precision {} Recall {}'.format(pre, rec))
    beta = 1
    f1score = 0
    if pre != 0 or rec != 0:
        f1score = (beta * beta + 1) * pre * rec / \
                  (beta * beta * pre + rec)
    logging.info('F1 {}'.format(f1score))
    return f1score


# Compute f-score
# TODO use masks to remove the remaining loop
# TODO can run this vectorized function on the GPU for faster performance
def compute_fscore_vectorized(Y_pred, Y_true):
    pre = 0
    pre_tot = 0
    rec = 0
    rec_tot = 0
    corr = 0
    total = 0
    number_of_examples = len(Y_true)
    for i in range(number_of_examples):
        # TODO path sentence length as a third parameter
        sentence_length = len(Y_true[i])
        true_array = Y_true[i]
        pred_array = Y_pred[i][:sentence_length]
        is_correct = (pred_array == true_array)
        is_positive = (true_array != helpers.labels_map['O'])
        corr += np.sum(is_correct)
        rec_tot += np.sum(is_positive)
        rec += np.sum(is_positive & is_correct)
        is_predicted = pred_array != helpers.labels_map['O']
        pre_tot += np.sum(is_predicted)
        pre += np.sum(is_predicted & is_correct)
        total += sentence_length
    res = corr * 1. / total
    logging.info('Accuracy (token level) {}'.format(res))
    if pre_tot == 0:
        pre = 0
    else:
        pre = 1. * pre / pre_tot
    rec = 1. * rec / rec_tot
    logging.info('Precision {} Recall {}'.format(pre, rec))
    beta = 1
    f1score = 0
    if pre != 0 or rec != 0:
        f1score = (beta * beta + 1) * pre * rec / \
                  (beta * beta * pre + rec)
    logging.info('F1 {}'.format(f1score))
    return f1score


def build_keras_model(max_len, input_dim, output_dim, embedding_matrix):
    logging.info('building Keras model...')
    input = Input(shape=(max_len,))
    # TODO use fixed embeddings
    model = Embedding(input_dim=input_dim, output_dim=output_dim, input_length=max_len,
                      weights=[embedding_matrix], trainable=False)(input)
    model = Dropout(0.1)(model)
    n_units = 128
#    model = LSTM(units=n_units, return_sequences=True, recurrent_dropout=0.1)(model)
    # TODO the model below is a stronger bi-directional model
    model = Bidirectional(LSTM(units=n_units, return_sequences=True, recurrent_dropout=0.1))(model)
    # TODO this should be a parameter
    n_tags = 5
    # TODO what does TimeDistributed do?
    out = TimeDistributed(Dense(n_tags, activation='softmax'))(model)
    model = Model(input, out)
    logging.info('Model type: ')
    logging.info(type(model))
    logging.info('Model summary: ')
    logging.info(model.summary())
    logging.info('done building model...')
    return model


# TODO Implement RNN model
# TODO handle off by one in tags
class RNNTagger(object):
    def __init__(self, model_file, max_len, input_dim, output_dim, embedding_matrix):
        logging.info('RNN Tagger')
        self.model_file = model_file
        self.name = 'RNN'
        # TODO handle untrained model prediction / confidence
        self.model = build_keras_model(max_len=max_len, input_dim=input_dim, output_dim=output_dim,
                                       embedding_matrix=embedding_matrix)
        # TODO how many times should we compile the model?
        # TODO optimize the learning rate parameters for this model!
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        self.max_len = max_len

    # TODO Avoid lists all together by allocating large ndarray objects
    # TODO Fine tune instead of learn from scratch
    # TODO fix the word embeddings
    # TODO test offfline
    def train(self, idx, idy):
        logging.info('starting training...')
        # TODO can we avoid this np.array call?
        idx = np.array(idx)
        idy = np.array(idy)
        # TODO do we need a validation split?
        # TODO these should be parameters
        self.model.fit(idx, idy, batch_size=200, epochs=20, verbose=1)
        logging.info('done training...')

    # TODO use word embeddings
    # TODO label should be an ndarray
    # Keep track of length mask
    def test(self, features, labels):
        assert self.model is not None
        logging.info('starting testing...')
        # TODO what is the batch size?
        # Forward prop to get the predictions
        num_samples = features.shape[0]
        logging.info('Number of samples: {}'.format(num_samples))
        max_batch_size = 4096
        batch_size = min(num_samples, max_batch_size)
        predictions_probability = self.model.predict(features, batch_size=batch_size)
        predictions = np.argmax(predictions_probability, axis=-1)
        fscore = compute_fscore(Y_pred=predictions, Y_true=labels)
        logging.info('done testing...')
        return fscore

    def get_confidence(self, sentence_idx):
        if self.model is None:
            # We haven't trained anything yet!
            return [0.2]
        else:
            # TODO might need to tune the temperature parameter
            predictions_marginals = self.model.predict(sentence_idx.reshape(1, -1))
            predictions_probabilities = np.max(predictions_marginals, axis=-1)
            # TODO PAD should be a constant
            PAD = 0
            sentence_length = np.sum(sentence_idx != PAD)
            # Remove padding when computing the confidence
            predictions_probabilities = predictions_probabilities[:, :sentence_length]
            log_predictions_probabilities = np.log(predictions_probabilities)
            # TODO we're using log probabilities instead of probabilities
            # TODO we might have to tune the temperature parameter to get calibrated probabilities
            confidence = log_predictions_probabilities.sum() / sentence_length
            # TODO match probabilities from pycrfsuite
            # TODO compute probability of a sequence
            # TODO would it be better to compute the sum of log probabilities instead of the probabilities?!
            return [confidence]

    # TODO add assert statements
    def get_predictions(self, features):
        # TODO This should be a constant!
        PAD = 0
        sentence_length = np.sum(features != PAD)
        n_tags = 5
        if self.model is None:
            # TODO this should be a parameter
            # TODO can use numpy to make this faster
            y_marginals = np.ones((sentence_length, n_tags)) * 0.2
            return y_marginals
        else:
            predictions_marginals = self.model.predict(features.reshape(1, -1))
            # Flatten by removing the extra dimension, and remove padding
            predictions_marginals = predictions_marginals[0, :sentence_length, :]
#            y_marginals = np.ones((sentence_length, n_tags)) * 0.2
#            return y_marginals
            return predictions_marginals


# TODO abstract away what is common with RNN Tagger
class CRFTagger(object):

    def __init__(self, model_file):
        logging.info('CRF Tagger')
        self.model_file = model_file
        self.name = 'CRF'

    def train(self, train_sents):
        X_train = [sent2features(s) for s in train_sents]
        Y_train = [sent2labels(s) for s in train_sents]
        trainer = pycrfsuite.Trainer(verbose=False)
        for xseq, yseq in zip(X_train, Y_train):
            trainer.append(xseq, yseq)
        trainer.set_params({
            'c1': 1.0,   # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 50,  # stop earlier
            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })
        trainer.train(self.model_file)
        if len(trainer.logparser.iterations) != 0:
            logging.info('{} {}'.format(len(trainer.logparser.iterations), trainer.logparser.iterations[-1]))
        else:
            # TODO
            logging.info(len(trainer.logparser.iterations))
            logging.info('There is no loss to present')

    # different lens
    # TODO can we refactor the sent2features better to avoid similar bugs in the future?
    # TODO review this function to make sure it's doing the right thing!
    def get_predictions(self, sent):
        sent = sent.split()
        # use the same interface sent2features expects
        sent = [(s, '') for s in sent]
        sent = sent2features(sent)
        tagger = pycrfsuite.Tagger()
        if not os.path.isfile(self.model_file):
            y_marginals = []
            for i in range(len(sent)):
                y_marginals.append([0.2] * 5)
            return y_marginals
        tagger.open(self.model_file)
        tagger.set(sent)
        # TODO this should be a function
        y_marginals = []
        # print "Tagset", tagger.labels()
        # ['1', '2', '3', '4', '5']
        # if len(tagger.labels) < 5
        for i in range(len(sent)):
            y_i = []
            for y in range(1, 6):
                if str(y) in tagger.labels():
                    y_i.append(tagger.marginal(str(y), i))
                else:
                    y_i.append(0.)
            y_marginals.append(y_i)
        return y_marginals

    # use P(yseq|xseq)
    def get_confidence(self, sent):
        sent = sent.split()
        # Add a dummy label because sent2features using this interface
        sent = [(s, '') for s in sent]
        sent = sent2features(sent)
        tagger = pycrfsuite.Tagger()
        if not os.path.isfile(self.model_file):
            confidence = 0.2
            return [confidence]
        tagger.open(self.model_file)
        tagger.set(sent)
        Y_pred = tagger.tag()
        p_y_pred = tagger.probability(Y_pred)
        confidence = pow(p_y_pred, 1. / len(Y_pred))
        return [confidence]

    # TODO this seems to be unused?!
    def get_uncertainty(self, sent):
        sent = sent.split()
        # use the same interface sent2features expects
        sent = [(s, '') for s in sent]
        x = sent2features(sent)
        tagger = pycrfsuite.Tagger()
        if not os.path.isfile(self.model_file):
            unc = random.random()
            return unc
        tagger.open(self.model_file)
        tagger.set(x)
        ttk = 0.
        for i in range(len(x)):
            y_probs = []
            for y in range(1, 6):
                if str(y) in tagger.labels():
                    y_probs.append(tagger.marginal(str(y), i))
                else:
                    y_probs.append(0.)
            ent = 0.
            for y_i in y_probs:
                if y_i > 0:
                    ent -= y_i * math.log(y_i, 5)
            ttk += ent
        return ttk

    def test(self, X_test, Y_true):
        tagger = pycrfsuite.Tagger()
        tagger.open(self.model_file)
        # TODO this list comprehension is so slow, can we make it faster?
        Y_pred = [tagger.tag(xseq) for xseq in X_test]
        f1score = compute_fscore(Y_pred=Y_pred, Y_true=Y_true)
        return f1score
