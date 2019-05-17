import argparse
import logging
import os
import random as rn
import sys
import time
from collections import defaultdict
from itertools import chain
from collections import namedtuple

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import helpers
import tagger
from game_ner import NERGame
from robot import RobotCNNDQN
from robot import RobotRandom
from tagger import CRFTagger
from tagger import RNNTagger

Language = namedtuple('Language', ['train', 'test', 'dev', 'emb', 'tagger'])


# TODO call by reference global variables!
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', help="require a decision agent")
    parser.add_argument( '--episode', type=int, help="require a maximum number of playing the game")
    parser.add_argument('--budget', type=int, help="requrie a budget for annotating")
    parser.add_argument('--train', help="training phase")
    parser.add_argument('--test', help="testing phase")
    # tensorflow flag for the maximum sequence length
    parser.add_argument('--max_seq_len', type=int, default=120, required=False, help='sequence')
    # tensorflow flag for the maximum vocabulary size
    parser.add_argument('--max_vocab_size', type=int, default=20000, required=False, help='vocabulary')
    # Embedding size
    parser.add_argument('--embedding_size', type=int, default=40, required=False, help='embedding size')
    parser.add_argument('--log_path', type=str, required=False, default='log.txt', help='log file path')
    # Log level
    parser.add_argument('--log_level', type=str, required=False, default='INFO', help='logging level')
    # model name
    parser.add_argument('--model_name', type=str, default='CRF', help='model name')
    return parser.parse_args()


def assert_list_of_sentences(sentences):
    assert type(sentences) == list
    assert len(sentences) >= 1
    assert type(sentences[0]) == str


def get_word_frequencies(sentences):
    words_dict = defaultdict(lambda: 0)
    for sentence in sentences:
        for word in sentence.split():
            words_dict[word] = words_dict[word] + 1
    return words_dict


# Returns ndarray mapping every word to an index
# TODO can we start tags from zero for pycrfsuite
# TODO create a reverse index!
# TODO can we make this faster?
# TODO handle tags and padding for tags
# TODO also use the same function for padding tags
def sentences_to_idx(sentences, word_to_idx, max_len, pad_value, unk_value):
    unpadded_sequence = [[word_to_idx[w] if w in word_to_idx else unk_value for w in s.split()]for s in sentences]
    # pad_sequences return an ndarray which is exactly what we want
    padded_sequence = pad_sequences(maxlen=max_len, sequences=unpadded_sequence, padding='post', value=pad_value)
    return padded_sequence


def labels_to_idy(labels, max_len, num_tags):
    # TODO 4 should be a parameter
    padded_labels = pad_sequences(maxlen=max_len, sequences=labels, padding='post', value=4)
    # TODO stay in numpy land
    padded_labels = [to_categorical(i, num_classes=num_tags) for i in padded_labels]
    return np.array(padded_labels)


# Creates a vocabulary given iterator over sentences and maximum vocab size
def get_vocabulary(sentences, max_vocab_size):
    # assert max_vocab_size at least two
    assert max_vocab_size >= 2
    # Step 1 identify all unique words
    words = get_word_frequencies(sentences=sentences)
    logging.debug('number of unique words before frequency trimming: {}'.format(len(words)))
    # Sort by frequency
    sorted_words = sorted(words.items(), key=lambda x: -x[1])
    # Only create vocab for the max_vocab_size entries, we don't need frequency anymore
    sorted_words = [w for (w, f) in sorted_words[:max_vocab_size-2]]
    n_words = len(sorted_words)
    logging.debug('number of unique words after frequency trimming: {}'.format(n_words))
    # TODO define these as constants
    # pad by zeros
    pad_value = 0
    # unk by n_words + 1
    unk_value = n_words + 1
    word_to_idx = {}
    word_to_idx['UNK'] = unk_value
    word_to_idx['PAD'] = pad_value
    for i, w in enumerate(sorted_words):
        word_to_idx[w] = i + 1
    return word_to_idx


def setup_tensorflow_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # TODO can run faster by not logging this
    # TODO create a command line flag for this
#    config.log_device_placement = True
    return tf.Session(config=config)


# TODO is there a better way to handle UNK and PAD?
# TODO also keep track of train_idy, a padded version of tags
def initialize_game(train_file, test_file, dev_file, emb_file, budget, max_seq_len, max_vocab_size, emb_size,
                    model_name):
    # Load data
    logging.debug('Loading data ..')
    # TODO utilize train_lens, test_lens, dev_lens
    train_x, train_y, train_lens = helpers.load_data2labels(input_file=train_file, max_len=max_seq_len)
    test_x, test_y, test_lens = helpers.load_data2labels(input_file=test_file, max_len=max_seq_len)
    dev_x, dev_y, dev_lens = helpers.load_data2labels(input_file=dev_file, max_len=max_seq_len)
    logging.debug('Processing data')
    # Build vocabulary
    logging.debug('Max document length: {}'.format(max_seq_len))
    # Create vocabulary
    word_to_idx = get_vocabulary(sentences=chain(train_x, dev_x, test_x), max_vocab_size=max_vocab_size)
    pad_value = word_to_idx['PAD']
    unk_value = word_to_idx['UNK']
    # Train
    train_idx = sentences_to_idx(sentences=train_x, word_to_idx=word_to_idx, max_len=max_seq_len, pad_value=pad_value,
                                 unk_value=unk_value)
    # TODO this should be a parameter
    num_tags = 5
    train_idy = labels_to_idy(labels=train_y, max_len=max_seq_len, num_tags=num_tags)
    # Dev
    dev_idx = sentences_to_idx(sentences=dev_x, word_to_idx=word_to_idx, max_len=max_seq_len, pad_value=pad_value,
                               unk_value=unk_value)
    dev_idy = labels_to_idy(labels=dev_y, max_len=max_seq_len, num_tags=num_tags)
    # Test
    test_idx = sentences_to_idx(sentences=test_x, word_to_idx=word_to_idx, max_len=max_seq_len, pad_value=pad_value,
                                unk_value=unk_value)
    test_idy = labels_to_idy(labels=test_y, max_len=max_seq_len, num_tags=num_tags)
    # Build embeddings
    w2v = helpers.load_crosslingual_embeddings(input_file=emb_file, vocab=word_to_idx, max_vocab_size=max_vocab_size,
                                               emb_size=emb_size)
    # prepare story
    story = [train_x, train_y, train_idx, train_idy]
    logging.info('The length of the story {} (DEV = {}  TEST = {})'.format(len(train_x), len(dev_x), len(test_x)))
    test = [test_x, test_y, test_idx, test_idy]
    dev = [dev_x, dev_y, dev_idx, dev_idy]
    # load game
    logging.info('Loading game ..')
    # TODO use named arguments here
    game = NERGame(story=story, dev=dev, max_len=max_seq_len, w2v=w2v, budget=budget, model_name=model_name)
    return game


def test_agent_batch(robot, game, model, budget):
    i = 0
    queried_x = []
    queried_y = []
    performance = []
    test_sents = helpers.data2sents(game.test_x, game.test_y)
    X_test = [tagger.sent2features(s) for s in test_sents]
    Y_true = [tagger.sent2labels(s) for s in test_sents]
    game.reboot(model)
    while i < budget:
        sel_ind = game.current_frame
        # construct the observation
        observation = game.get_frame(model)
        action = robot.get_action(observation)
        if action[1] == 1:
            sentence = game.train_x[sel_ind]
            labels = game.train_y[sel_ind]
            queried_x.append(sentence)
            queried_y.append(labels)
            i += 1
            train_sents = helpers.data2sents(queried_x, queried_y)
            model.train(train_sents)
            performance.append(model.test(X_test, Y_true))
        game.current_frame += 1
    # train a crf and evaluate it
    train_sents = helpers.data2sents(queried_x, queried_y)
    model.train(train_sents)
    performance.append(model.test(X_test, Y_true))
    logging.debug('***TEST {}'.format(performance))


def test_agent_online(robot, game, model, budget):
    # to address game -> we have a new game here
    i = 0
    queried_x = []
    queried_y = []
    performance = []
    test_sents = helpers.data2sents(game.test_x, game.test_y)
    X_test = [tagger.sent2features(s) for s in test_sents]
    Y_true = [tagger.sent2labels(s) for s in test_sents]
    game.reboot(model)
    while i < budget:
        sel_ind = game.current_frame
        # construct the observation
        observation = game.get_frame(model)
        action = robot.get_action(observation)
        if action[1] == 1:
            sentence = game.train_x[sel_ind]
            labels = game.train_y[sel_ind]
            queried_x.append(sentence)
            queried_y.append(labels)
            i += 1
            train_sents = helpers.data2sents(queried_x, queried_y)
            model.train(train_sents)
            performance.append(model.test(X_test, Y_true))
        reward, observation2, terminal = game.feedback(action, model)  # game
        robot.update(observation, action, reward, observation2, terminal)
    # train a crf and evaluate it
    train_sents = helpers.data2sents(queried_x, queried_y)
    model.train(train_sents)
    performance.append(model.test(X_test, Y_true))
    logging.debug('***TEST {}'.format(performance))


def build_model(model_name, model_file, max_len, input_dim, output_dim, embedding_matrix):
    if model_name == 'CRF':
        model = CRFTagger(model_file=model_file)
    elif model_name == 'RNN':
        model = RNNTagger(model_file=model_file, max_len=max_len, input_dim=input_dim, output_dim=output_dim,
                          embedding_matrix=embedding_matrix)
    else:
        logging.error('Invalid model type')
        assert False
    return model


def play_ner(agent, train_lang, budget, max_seq_len, max_vocab_size, embedding_size, max_episode, emb_size, model_name,
             session):
    train_lang_num = len(train_lang)
    actions = 2
    if agent == 'random':
        logging.info('Creating random robot...')
        robot = RobotRandom(actions)
    elif agent == 'dqn':
        # TODO Implement this
        assert False
#        robot = RobotDQN(actions)
    elif agent == 'cnndqn':
        logging.info('Creating CNN DQN robot...')
        robot = RobotCNNDQN(actions, embedding_size=embedding_size, max_len=max_seq_len, session=session)
    else:
        logging.info('** There is no robot.')
        raise SystemExit

    for i in range(train_lang_num):
        train = train_lang[i].train
        test = train_lang[i].test
        dev = train_lang[i].dev
        emb = train_lang[i].emb
        model_file = train_lang[i].tagger
        # initialize a NER game
        game = initialize_game(train, test, dev, emb, budget, max_seq_len=max_seq_len, max_vocab_size=max_vocab_size,
                               emb_size=emb_size, model_name=model_name)
        # initialize a decision robot
        # robot.initialise(game.max_len, game.w2v)
        robot.update_embeddings(game.w2v)
        # tagger
        model = build_model(model_name=model_name, model_file=model_file, max_len=max_seq_len, input_dim=max_vocab_size,
                            output_dim=emb_size, embedding_matrix=game.w2v)
        # play game
        episode = 1
        logging.info('>>>>>> Playing game ..')
        gamma = 0.99
        episode_return = 0.0
        total_episodic_return = 0.0
        episode_start = time.clock()
        total_episodic_time = 0
        while episode <= max_episode:
            observation = game.get_frame(model)
            action = robot.get_action(observation)
            logging.debug('> Action {}'.format(action))
            reward, observation2, terminal = game.feedback(action, model)
            episode_return = gamma * episode_return + reward
            logging.debug('> Reward {}'.format(reward))
            robot.update(observation, action, reward, observation2, terminal)
            if terminal:
                total_episodic_return = total_episodic_return + episode_return
                average_episodic_return = total_episodic_return / float(episode + 1)
                episode_time = time.clock() - episode_start
                total_episodic_time = total_episodic_time + episode_time
                logging.info('>>>>>>> {0} / {1} episode return: {2:.4f}, average return: {3:.4f}, episode time: {4:.4f}s, total time: {5:4f}s, last f-score: {6:.4}'.format(
                    episode, max_episode, episode_return, average_episodic_return, episode_time, total_episodic_time, game.performance))
                # Reset return and time for next episode
                episode += 1
                episode_return = 0.0
                episode_start = time.clock()
    return robot


def run_test(robot, test_lang, budget, max_seq_len, max_vocab_size, emb_size, model_name):
    test_lang_num = len(test_lang)
    # TODO can do more pythonic looping
    for i in range(test_lang_num):
        train = test_lang[i].train
        test = test_lang[i].test
        dev = test_lang[i].dev
        emb = test_lang[i].emb
        model_file = test_lang[i].tagger
        game2 = initialize_game(train, test, dev, emb, budget, max_seq_len=max_seq_len, max_vocab_size=max_vocab_size,
                                emb_size=emb_size, model_name=model_name)
        robot.update_embeddings(game2.w2v)
        model = build_model(model_name=model_name, model_file=model_file, input_dim=max_vocab_size, output_dim=emb_size,
                            embedding_matrix=game2.w2v)
        test_agent_batch(robot, game2, model, budget)
        test_agent_online(robot, game2, model, budget)


# TODO can we get rid of the side effects in this function?
def set_logger(log_path, log_level):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)
    # Log to console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)


def construct_languages(all_langs):
    # load the train data: source languages
    parts = all_langs.split(',')
    train_lang_num = int(len(parts) / 5)
    if len(parts) % 5 != 0:
        logging.debug('Wrong inputs of training')
        raise SystemExit
    langs = []
    for i in range(train_lang_num):
        lang_i = i * 5
        train = parts[lang_i + 0]
        test = parts[lang_i + 1]
        dev = parts[lang_i + 2]
        emb = parts[lang_i + 3]
        tagger = parts[lang_i + 4]
        langs.append(Language(train=train, test=test, dev=dev, emb=emb, tagger=tagger))
    return langs


def fix_random_seeds():
    # fix random seed for numpy
    np.random.seed(42)
    # fix random seed for python random module
    rn.seed(12345)
    # fix random seed for tensorflow backend
    tf.set_random_seed(1234)


def main():
    args = parse_args()
    set_logger(args.log_path, args.log_level)
    logging.debug('working directory: {}'.format(os.getcwd()))
    logging.debug('got args: ')
    logging.debug(args)
    logging.debug('fixing random seed, for full reproducibility, run on CPU and turn off multi-thread operations...')
    fix_random_seeds()
    budget = args.budget
    train_lang = construct_languages(args.train)
    # load the test data: target languages
    test_lang = construct_languages(args.test)
    max_seq_len = args.max_seq_len
    max_vocab_size = args.max_vocab_size
    embedding_size = args.embedding_size
    model_name = args.model_name
    logging.debug('setting up tensorflow and keras sessions...')
    session = setup_tensorflow_session()
    K.set_session(session)
    logging.debug('done setting up keras session...')
    # play games for training a robot
    robot = play_ner(agent=args.agent, train_lang=train_lang, budget=budget, max_seq_len=max_seq_len,
                     max_vocab_size=max_vocab_size, embedding_size=embedding_size,
                     max_episode=args.episode, emb_size=embedding_size, model_name=model_name, session=session)
    # play a new game with the trained robot
    run_test(robot=robot, test_lang=test_lang, budget=budget, max_seq_len=max_seq_len, max_vocab_size=max_vocab_size,
             emb_size=embedding_size, model_name=model_name)


if __name__ == '__main__':
    main()
