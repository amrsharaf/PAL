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

# TODO fix this
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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


# Sequence to sequence model
# TODO why are the labels not used?
# What is mode?
# params: contains hyper-parameters
# TODO seems like the evaluation mode is not implemented!
def seq2seq_model(features, labels, mode, params):
    ops = {}
    # Training mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        batch_sz = tf.shape(features['input'])[0]
        with tf.variable_scope('main', reuse=False):
            embedding = tf.get_variable('lookup_table', [params['vocab_size'], params['hidden_dim']])
#            cells = multi_cell_fn()
            # TODO Implement this
            cells = None
            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=tf.nn.embedding_lookup(embedding, features['input']),
                sequence_length=tf.count_nonzero(features['input'], 1, dtype=tf.int32))
            # TODO replace the decoder with a simple classifier
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=cells,
                helper=helper,
                initial_state=cells.zero_state(batch_sz, tf.float32),
                output_layer=tf.layers.Dense(params['vocab_size']))
            decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder)
            logits = decoder_output.rnn_output
            output = features['output']
            ops['global_step'] = tf.Variable(0, trainable=False)
            ops['loss'] = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(
                logits=logits, targets=output, weights=tf.to_float(tf.ones_like(output))))
            # TODO implement clip_grads
            ops['train'] = tf.train.AdamOptimizer().apply_gradients(
                (ops['loss']), global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=ops['loss'], train_op=ops['train'])
    # Prediction mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        with tf.variable_scope('main', reuse=True):
#            cells = multi_cell_fn()
            cells = None
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=cells,
                embedding=tf.get_variable('lookup_table'),
                start_tokens=tf.tile(tf.constant([params['char2idx']['<start>']], dtype=tf.int32), [1]),
                end_token=params['char2idx']['<end>'],
                initial_state=tf.contrib.seq2seq.tile_batch(cells.zero_state(1, tf.float32), params['beam_width']),
                beam_width=params['beam_width'],
                output_layer=tf.layers.Dense(params['vocab_size'], _reuse=True))
            decoder_out, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, maximum_iterations=params['seq_len'])
            tf.identity(decoder_out[0].predicted_ids, name='predictions')
            predictions = decoder_out.predicted_ids[:, :, 0]
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


# TODO Implement RNN model
class RNNTagger(object):
    def __init__(self, model_file):
        logging.info('RNN Tagger')
        self.model_file = model_file
        self.name = 'RNN'
        # Untrained model
        self.model = None

    def train(self, idx, y):
        logging.info('starting training...')
        # keras implementation
        # TODO this should be a parameter
        max_len = 120
        input = Input(shape=(max_len,))
        # TODO this should be a parameter
        n_words = 20000
        model = Embedding(input_dim=n_words, output_dim=40, input_length=max_len)(input)
        model = Dropout(0.1)(model)
        n_units = 128
        model = Bidirectional(LSTM(units=n_units, return_sequences=True, recurrent_dropout=0.1))(model)
        n_tags = 5
        out = TimeDistributed(Dense(n_tags, activation='softmax'))(model)
        self.model = Model(input, out)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics='accuracy')
        # TODO do we need a validation split?
        self.model.fit(idx, y, batch_size=32, epochs=5, verbose=1)
        # After defining the model, we run training steps by passing in batched inputs. we use TensorFlow Estimator API
        # to train the model.
        # TODO set model_dir to avoid the creation of a temporary directory
#        estimator = tf.estimator.Estimator(model_fn=seq2seq_model, params=params)
#        estimator.train(input_fn=lambda: input_fn(ints), steps=1000)
#        estimator = tf.estimator.Estimator(model_fn=seq2seq_model, model_dir='model_dir', params=params)
        assert False

    # TODO use word embeddings
    def test(self, features, labels):
        print('inside test...')
        print('got features: ', features)
        print('labels: ', labels)
        # Forward prop to get the predictions
        if self.model is None:
            return None
        else:
            predictions = self.model.predict(features)
        assert False

    def get_confidence(self, sentence_idx):
        if self.model is None:
            # We haven't trained anything yet!
            return [0.2]
        else:
            print('got sentence_idx: ', sentence_idx)
            # TODO do we need to load a model from a file every time we make a prediction?
            prediction = self.test(idx=sentence_idx, y=None)
            print('got prediction: ', prediction)
            assert False
#        sent = sent2features(sent)
#        tagger = pycrfsuite.Tagger()
#        if not os.path.isfile(self.model_file):
#            confidence = 0.2
#            return [confidence]
#        tagger.open(self.model_file)
#        tagger.set(sent)
#        y_pred = tagger.tag()
#        p_y_pred = tagger.probability(y_pred)
#        confidence = pow(p_y_pred, 1. / len(y_pred))
#        return [confidence]

    def get_predictions(self, features):
        print('got features: ', features)
        if self.model is None:
            print('we have not trained anything yet')
            y_marginals = []
            # TODO compute sentence length correctly
            # TODO replace zero with the padding token
            sentence_length = np.sum(features != 0)
            # TODO this should be a parameter
            n_tags = 5
            # TODO can use numpy to make this faster
            for i in range(sentence_length):
                y_i = []
                for y in range(n_tags):
                    y_i.append(0.2)
                y_marginals.append(y_i)
            return y_marginals
        else:
            print('the model has been trained!')
        assert False


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
        y_pred = tagger.tag()
        p_y_pred = tagger.probability(y_pred)
        confidence = pow(p_y_pred, 1. / len(y_pred))
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
        y_pred = [tagger.tag(xseq) for xseq in X_test]
        pre = 0
        pre_tot = 0
        rec = 0
        rec_tot = 0
        corr = 0
        total = 0
        for i in range(len(Y_true)):
            for j in range(len(Y_true[i])):
                total += 1
                if y_pred[i][j] == Y_true[i][j]:
                    corr += 1
                if helpers.label2str[int(y_pred[i][j])] != 'O':  # not 'O'
                    pre_tot += 1
                    if y_pred[i][j] == Y_true[i][j]:
                        pre += 1
                if helpers.label2str[int(Y_true[i][j])] != 'O':
                    rec_tot += 1
                    if y_pred[i][j] == Y_true[i][j]:
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
