# TODO better import structure
from launcher_ner_bilingual import initialize_game
from launcher_ner_bilingual import parse_args
from launcher_ner_bilingual import construct_languages
from launcher_ner_bilingual import set_logger
import logging
from keras.layers import Embedding, Dropout, LSTM, Bidirectional, TimeDistributed, Dense
from keras.models import Input, Model
from hyperas import optim
from hyperopt import Trials, tpe
from tagger import compute_fscore
import numpy as np
from hyperas.distributions import uniform


# TODO avoid global variables
def data():
    global x_train_global, y_train_global, x_test_global, y_test_global
    # TODO limit by budget
    return x_train_global, y_train_global, x_test_global, y_test_global


# TODO can we avoid copy-paste for the kears model?
# TODO can we avoid global variables?
def create_model(x_train, y_train, x_test, y_test):
    global max_len, input_dim, embedding_matrix, output_dim
    logging.debug('building Keras model...')
    input = Input(shape=(max_len,))
    model = Embedding(input_dim=input_dim, output_dim=output_dim, input_length=max_len,
                      weights=[embedding_matrix], trainable=False)(input)
    model = Dropout({{uniform(0, 1)}})(model)
    n_units = 128
    #    model = Dense(n_units)(model)
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
    logging.info('starting training...')
    # Train only on the last history
    model.fit(x_train, y_train, batch_size=200, epochs=20, verbose=0)
    logging.info('done training...')
    logging.info('starting testing...')
    # Forward prop to get the predictions
    num_samples = x_test.shape[0]
    logging.info('Number of samples: {}'.format(num_samples))
    max_batch_size = 4096
    batch_size = min(num_samples, max_batch_size)
    predictions_probability = model.predict(x_test, batch_size=batch_size)
    predictions = np.argmax(predictions_probability, axis=-1)
    fscore = compute_fscore(Y_pred=predictions, Y_true=y_test)
    logging.info('done testing...')
    return -1.0 * fscore


def main():
    global max_len, input_dim, embedding_matrix, output_dim
    global x_train_global, y_train_global, x_test_global, y_test_global
    args = parse_args()
    set_logger(args.log_path, args.log_level)
    logging.info('Args:')
    logging.info(args)
    lang = construct_languages(args.train)
    assert len(lang) == 1
    lang = lang[0]
    # TODO Initializing a full game is an over-kill, maybe expose an easier interface for creating data
    game = initialize_game(train_file=lang.train, test_file=lang.test, dev_file=lang.dev, emb_file=lang.emb,
                           budget=args.budget, max_seq_len=args.max_seq_len, max_vocab_size=args.max_vocab_size,
                           emb_size=args.embedding_size, model_name=args.model_name)
    x_train_global = game.train_idx
    y_train_global = game.train_y
    x_test_global = game.dev_idx
    y_test_global = game.dev_y
    # TODO fix this interface
    max_len = args.max_seq_len
    input_dim = args.max_vocab_size
    output_dim = args.embedding_size
    embedding_matrix = game.w2v
    logging.info('done creating data and model objects')
    best_run, best_model = optim.minimize(model=create_model, data=data, algo=tpe.suggest, max_evals=5, trials=Trials())
    x_train, y_train, x_test, y_test = data()
    logging.info('Evaluation of best performing model:')
    logging.info(best_model.evaluate(x_test, y_test))
    logging.info('Best performing model chosen hyper-parameters:')
    logging.info(best_run)


if __name__ == '__main__':
    main()
