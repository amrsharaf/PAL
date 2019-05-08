import sys
import argparse
from game_ner import NERGame
from robot import RobotCNNDQN
import numpy as np
import helpers
import tensorflow as tf
from tagger import CRFTagger
from tagger import RNNTagger
import logging
import tagger
import os


# TODO call by reference global variables!
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', help="require a decision agent")
    parser.add_argument( '--episode', type=int, help="require a maximum number of playing the game")
    parser.add_argument('--budget', help="requrie a budget for annotating")
    parser.add_argument('--train', help="training phase")
    parser.add_argument('--test', help="testing phase")
    # tensorflow flag for the maximum sequence length
    parser.add_argument('--max_seq_len', default=120, required=False, help='sequence')
    # tensorflow flag for the maximum vocabulary size
    parser.add_argument('--max_vocab_size', default=20000, required=False, help='vocabulary')
    # Embedding size
    parser.add_argument('--embedding_size', type=int, default=40, required=False, help='embedding size')
    parser.add_argument('--log_path', type=str, required=False, default='log.txt', help='log file path')
    # model name
    parser.add_argument('--model_name', type=str, default='CRF', help='model name')
    return parser.parse_args()


def assert_list_of_sentences(sentences):
    assert type(sentences) == list
    assert len(sentences) >= 1
    assert type(sentences[0]) == str


def get_unique_words(sentences):
    words_set = set()
    for sentence in sentences:
        for word in sentence.split():
            words_set.add(word)
    return words_set


def sentences_to_idx(sentences, word_to_idx):
    return [[word_to_idx[w] for w in s.split()]for s in sentences]


# Returns ndarray mapping every word to an index
# TODO handle unk, for now we assume we know the full vocab, so no unk?
def process_vocabulary(train_sentences, dev_sentences, test_sentences):
    # assert correct train data
    assert_list_of_sentences(train_sentences)
    # assert correct dev data
    assert_list_of_sentences(dev_sentences)
    # assert correct test data
    assert_list_of_sentences(test_sentences)
    # Step 1 identify all unique words
    words = get_unique_words(train_sentences).union(get_unique_words(dev_sentences)).union(
        get_unique_words(test_sentences))
    n_words = len(words)
    logging.info('number of unique words: ', n_words)
    word_to_idx = {w: i+1 for i, w in enumerate(words)}
    # TODO can we make this faster?
    train_idx = sentences_to_idx(sentences=train_sentences, word_to_idx=word_to_idx)
    dev_idx = sentences_to_idx(sentences=dev_sentences, word_to_idx=word_to_idx)
    test_idx = sentences_to_idx(sentences=test_sentences, word_to_idx=word_to_idx)
    # TODO handle padding
    assert False
    return train_idx, dev_idx, test_idx


def initialize_game(train_file, test_file, dev_file, emb_file, budget, max_seq_len, max_vocab_size, emb_size):
    # Load data
    logging.info('Loading data ..')
    train_x, train_y, train_lens = helpers.load_data2labels(train_file)
    test_x, test_y, test_lens = helpers.load_data2labels(test_file)
    dev_x, dev_y, dev_lens = helpers.load_data2labels(dev_file)
    logging.info('Processing data')
    # build vocabulary
    max_len = max_seq_len
    logging.info('Max document length: {}'.format(max_len))
    # vocab = vocab_processor.vocabulary_ # start from {"<UNK>":0}
    train_idx, dev_idx, test_idx = process_vocabulary(train_sentences=train_x, dev_sentences=dev_x,
                                                      test_sentences=test_x)
#    train_idx = np.array(list(vocab_processor.fit_transform(train_x)))
#    dev_idx = np.array(list(vocab_processor.fit_transform(dev_x)))
#    vocab = vocab_processor.vocabulary_
#    vocab.freeze()
#    test_idx = np.array(list(vocab_processor.fit_transform(test_x)))
#    # build embeddings
#    vocab = vocab_processor.vocabulary_
#    vocab_size = max_vocab_size
    w2v = helpers.load_crosslingual_embeddings(emb_file, vocab, vocab_size, emb_size=emb_size)
    # prepare story
    story = [train_x, train_y, train_idx]
    logging.info('The length of the story {0} (DEV = {1}  TEST = {2})'.format(len(train_x), len(dev_x), len(test_x)))
    test = [test_x, test_y, test_idx]
    dev = [dev_x, dev_y, dev_idx]
    # load game
    logging.info('Loading game ..')
    game = NERGame(story, test, dev, max_len, w2v, budget)
    return game


def test_agent_batch(robot, game, model, budget):
    i = 0
    queried_x = []
    queried_y = []
    performance = []
    test_sents = helpers.data2sents(game.test_x, game.test_y)
    X_test = [tagger.sent2features(s) for s in test_sents]
    Y_true = [tagger.sent2labels(s) for s in test_sents]
    game.reboot()
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
    logging.info('***TEST {0}'.format(performance))


def test_agent_online(robot, game, model, budget):
    # to address game -> we have a new game here
    i = 0
    queried_x = []
    queried_y = []
    performance = []
    test_sents = helpers.data2sents(game.test_x, game.test_y)
    X_test = [tagger.sent2features(s) for s in test_sents]
    Y_true = [tagger.sent2labels(s) for s in test_sents]
    game.reboot()
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
    logging.info('***TEST {0}'.format(performance))


def build_model(model_name, model_file):
    if model_name == 'CRF':
        model = CRFTagger(model_file=model_file)
    elif model_name == 'RNN':
        # TODO what is tagger?!
        model = RNNTagger(model_file=model_file)
    else:
        logging.error('Invalid model type')
        assert False
    return model


def play_ner(agent, train_lang, train_lang_num, budget, max_seq_len, max_vocab_size, embedding_size, max_episode,
             emb_size, model_name):
    actions = 2
    if agent == 'random':
        # TODO Implement this
        assert False
#        robot = RobotRandom(actions)
    elif agent == 'DQN':
        # TODO Implement this
        assert False
#        robot = RobotDQN(actions)
    elif agent == 'CNNDQN':
        robot = RobotCNNDQN(actions, embedding_size=embedding_size)
    else:
        logging.info('** There is no robot.')
        raise SystemExit

    for i in range(train_lang_num):
        train = train_lang[i][0]
        test = train_lang[i][1]
        dev = train_lang[i][2]
        emb = train_lang[i][3]
        model_file = train_lang[i][4]
        # initialize a NER game
        game = initialize_game(train, test, dev, emb, budget, max_seq_len=max_seq_len, max_vocab_size=max_vocab_size,
                               emb_size=emb_size)
        # initialise a decision robot
        # robot.initialise(game.max_len, game.w2v)
        robot.update_embeddings(game.w2v)
        # tagger
        model = build_model(model_name=model_name, model_file=model_file)
        # play game
        episode = 1
        logging.info('>>>>>> Playing game ..')
        while episode <= max_episode:
            logging.info('>>>>>>> Current game round {} Maximum  {}'.format(episode, max_episode))
            observation = game.get_frame(model)
            action = robot.get_action(observation)
            logging.info('> Action {}'.format(action))
            reward, observation2, terminal = game.feedback(action, model)
            logging.info('> Reward {}'.format(reward))
            robot.update(observation, action, reward, observation2, terminal)
            if terminal == True:
                episode += 1
                logging.info('> Terminal <')
    return robot


def run_test(robot, test_lang, test_lang_num, budget, max_seq_len, max_vocab_size, emb_size, model_name):
    for i in range(test_lang_num):
        train = test_lang[i][0]
        test = test_lang[i][1]
        dev = test_lang[i][2]
        emb = test_lang[i][3]
        model_file = test_lang[i][4]
        game2 = initialize_game(train, test, dev, emb, budget, max_seq_len=max_seq_len, max_vocab_size=max_vocab_size,
                                emb_size=emb_size)
        robot.update_embeddings(game2.w2v)
        model = build_model(model_name=model_name, model_file=model_file)
        test_agent_batch(robot, game2, model, budget)
        test_agent_online(robot, game2, model, budget)


# TODO can we get rid of the side effects in this function?
def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
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
        logging.info('Wrong inputs of training')
        raise SystemExit
    langs = []
    for i in range(train_lang_num):
        lang_i = i * 5
        train = parts[lang_i + 0]
        test = parts[lang_i + 1]
        dev = parts[lang_i + 2]
        emb = parts[lang_i + 3]
        tagger = parts[lang_i + 4]
        langs.append((train, test, dev, emb, tagger))
    return langs


def main():
    args = parse_args()
    logging.info('got args: ')
    logging.info(args)
    print('args: ', args)
    set_logger(args.log_path)
    budget = int(args.budget)
    train_lang = construct_languages(args.train)
    # load the test data: target languages
    test_lang = construct_languages(args.test)
    max_seq_len = args.max_seq_len
    max_vocab_size = args.max_vocab_size
    embedding_size = args.embedding_size
    model_name = args.model_name
    # play games for training a robot
    robot = play_ner(agent=args.agent, train_lang=train_lang, train_lang_num=len(train_lang), budget=budget,
                     max_seq_len=max_seq_len, max_vocab_size=max_vocab_size, embedding_size=embedding_size,
                     max_episode=args.episode, emb_size=embedding_size, model_name=model_name)
    # play a new game with the trained robot
    run_test(robot=robot, test_lang=test_lang, test_lang_num=len(test_lang), budget=budget, max_seq_len=max_seq_len,
             max_vocab_size=max_vocab_size, emb_size=embedding_size, model_name=model_name)


if __name__ == '__main__':
    print('working directory: ', os.getcwd())
    main()
