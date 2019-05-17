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


# Create data for hyper-parameter tuning
class Data(object):
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def __call__(self):
        # TODO limit by budget
        return self.x_train, self.y_train, self.x_test, self.y_test


# TODO can we avoid copy-paste for the kears model?
class CreateModel(object):
    def __init__(self, max_len, input_dim, output_dim, embedding_matrix):
        self.max_len = max_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_matrix = embedding_matrix

    def __call__(self, x_train, y_train, x_test, y_test):
        logging.debug('building Keras model...')
        input = Input(shape=(self.max_len,))
        model = Embedding(input_dim=self.input_dim, output_dim=self.output_dim, input_length=self.max_len,
                          weights=[self.embedding_matrix], trainable=False)(input)
        model = Dropout(0.1)(model)
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
        logging.debug('Model type: ')
        logging.debug(type(model))
        logging.debug('Model summary: ')
        logging.debug(model.summary())
        logging.debug('done building model...')
        return model


def main():
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
    x_train = game.train_idx
    y_train = game.train_y
    x_test = game.dev_idx
    y_test = game.dev_y
    data = Data(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    # TODO fix this interface
    create_model = CreateModel(max_len=args.max_seq_len, input_dim=args.max_vocab_size, output_dim=args.embedding_size,
                               embedding_matrix=game.w2v)
    logging.info('done creating data and model objects')
    best_run, best_model = optim.minimize(model=create_model, data=data, algo=tpe.suggest, max_evals=5, tirals=Trials())
    x_train, y_train, x_test, y_test =data()
    logging.info('Evaluation of best performing model:')
    logging.info(best_model.evaluate(x_test, y_test))
    logging.info('Best performing model chosen hyper-parameters:')
    logging.info(best_run)


if __name__ == '__main__':
    main()
