import logging

import keras
import numpy
from hyperas import optim
from hyperas.distributions import choice
from hyperopt import Trials, tpe
from keras.layers import Embedding, Dropout, LSTM, Bidirectional, TimeDistributed, Dense
from keras.models import Input, Model

from launcher_ner_bilingual import construct_languages
from launcher_ner_bilingual import initialize_game
from launcher_ner_bilingual import parse_args
from launcher_ner_bilingual import set_logger
from tagger import compute_fscore


def data():
    args = parse_args()
    set_logger(args.log_path, args.log_level)
    logging.debug('Args:')
    logging.debug(args)
    lang = construct_languages(args.train)
    assert len(lang) == 1
    lang = lang[0]
    game = initialize_game(train_file=lang.train, test_file=lang.test, dev_file=lang.dev, emb_file=lang.emb,
                           budget=args.budget, max_seq_len=args.max_seq_len, max_vocab_size=args.max_vocab_size,
                           emb_size=args.embedding_size, model_name=args.model_name)
    x_train = game.train_idx
    y_train = game.train_idy
    x_test = game.dev_idx
    y_test = game.dev_y
    permutation = numpy.random.permutation(len(x_train))
    x_train = x_train[permutation]
    y_train = y_train[permutation]
    x_train = x_train[:args.budget]
    y_train = y_train[:args.budget]
    return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):
    args = parse_args()
    set_logger(args.log_path, args.log_level)
    logging.debug('Args:')
    logging.debug(args)
    lang = construct_languages(args.train)
    assert len(lang) == 1
    lang = lang[0]
    game = initialize_game(train_file=lang.train, test_file=lang.test, dev_file=lang.dev, emb_file=lang.emb,
                           budget=args.budget, max_seq_len=args.max_seq_len, max_vocab_size=args.max_vocab_size,
                           emb_size=args.embedding_size, model_name=args.model_name)
    max_len = args.max_seq_len
    input_dim = args.max_vocab_size
    output_dim = args.embedding_size
    embedding_matrix = game.w2v
    logging.debug('building Keras model...')
    input = Input(shape=(max_len,))
    model = Embedding(input_dim=input_dim, output_dim=output_dim, input_length=max_len, weights=[embedding_matrix],
                      trainable=False)(input)
    model = Dropout(0.1)(model)
    n_units = 128
    model = Bidirectional(LSTM(units=n_units, return_sequences=True, recurrent_dropout=0.1))(model)
    n_tags = 5
    out = TimeDistributed(Dense(n_tags, activation='softmax'))(model)
    model = Model(input, out)
    logging.debug('Model type: ')
    logging.debug(type(model))
    logging.debug('Model summary: ')
    logging.debug(model.summary())
    rmsprop = keras.optimizers.RMSprop(lr={{choice([0.0001])}})
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
    logging.debug('done building model...')
    logging.debug('starting training...')
    num_train_examples = len(x_train)
    for i in range(num_train_examples):
        print('i: ', i)
        model.fit(x_train[:i], y_train[:i], batch_size=200, epochs=20, verbose=0)
    logging.debug('done training...')
    logging.debug('starting testing...')
    num_samples = x_test.shape[0]
    logging.debug('Number of samples: {}'.format(num_samples))
    max_batch_size = 4096
    batch_size = min(num_samples, max_batch_size)
    predictions_probability = model.predict(x_test, batch_size=batch_size)
    predictions = numpy.argmax(predictions_probability, axis=-1)
    fscore = compute_fscore(Y_pred=predictions, Y_true=y_test)
    logging.debug('done testing...')
    return -fscore


def main():
    best_run, best_model = optim.minimize(model=create_model, data=data, algo=tpe.suggest, max_evals=10, trials=Trials())
    logging.info('Best performing model chosen hyper-parameters:')
    logging.info(best_run)


if __name__ == '__main__':
    main()
