import sys
import argparse
from game_ner import NERGame
from robot import RobotCNNDQN
import numpy as np
import helpers
import tensorflow as tf
from tagger import CRFTagger
import logging


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
    return parser.parse_args()


def initialise_game(train_file, test_file, dev_file, emb_file, budget, max_seq_len, max_vocab_size):
    # Load data
    logging.info('Loading data ..')
    train_x, train_y, train_lens = helpers.load_data2labels(train_file)
    test_x, test_y, test_lens = helpers.load_data2labels(test_file)
    dev_x, dev_y, dev_lens = helpers.load_data2labels(dev_file)
    logging.info('Processing data')
    # build vocabulary
    max_len = max_seq_len
    logging.info('Max document length: {0}'.format(max_len))
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length=max_len, min_frequency=1)
    # vocab = vocab_processor.vocabulary_ # start from {"<UNK>":0}
    train_idx = np.array(list(vocab_processor.fit_transform(train_x)))
    dev_idx = np.array(list(vocab_processor.fit_transform(dev_x)))
    vocab = vocab_processor.vocabulary_
    vocab.freeze()
    test_idx = np.array(list(vocab_processor.fit_transform(test_x)))
    # build embeddings
    vocab = vocab_processor.vocabulary_
    vocab_size = max_vocab_size
    w2v = helpers.load_crosslingual_embeddings(emb_file, vocab, vocab_size)
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
            performance.append(model.test(test_sents))
        game.current_frame += 1
    # train a crf and evaluate it
    train_sents = helpers.data2sents(queried_x, queried_y)
    model.train(train_sents)
    performance.append(model.test(test_sents))
    logging.info('***TEST {0}'.format(performance))


def test_agent_online(robot, game, model, budget):
    # to address game -> we have a new game here
    i = 0
    queried_x = []
    queried_y = []
    performance = []
    test_sents = helpers.data2sents(game.test_x, game.test_y)
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
            performance.append(model.test(test_sents))
        reward, observation2, terminal = game.feedback(action, model)  # game
        robot.update(observation, action, reward, observation2, terminal)
    # train a crf and evaluate it
    train_sents = helpers.data2sents(queried_x, queried_y)
    model.train(train_sents)
    performance.append(model.test(test_sents))
    logging.info('***TEST {0}'.format(performance))


def play_ner(agent, train_lang, train_lang_num, budget, max_seq_len, max_vocab_size, embedding_size, max_episode):
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
        tagger = train_lang[i][4]
        # initialize a NER game
        game = initialise_game(train, test, dev, emb, budget, max_seq_len=max_seq_len, max_vocab_size=max_vocab_size)
        # initialise a decision robot
        # robot.initialise(game.max_len, game.w2v)
        robot.update_embeddings(game.w2v)
        # tagger
        model = CRFTagger(tagger)
        # play game
        episode = 1
        logging.info('>>>>>> Playing game ..')
        while episode <= max_episode:
            logging.info('>>>>>>> Current game round {0} Maximum  {1}'.format(episode, max_episode))
            observation = game.get_frame(model)
            action = robot.get_action(observation)
            logging.info('> Action {0}'.format(action))
            reward, observation2, terminal = game.feedback(action, model)
            logging.info('> Reward {0}'.format(reward))
            robot.update(observation, action, reward, observation2, terminal)
            if terminal == True:
                episode += 1
                logging.info('> Terminal <')
    return robot


def run_test(robot, test_lang, test_lang_num, budget, max_seq_len, max_vocab_size):
    for i in range(test_lang_num):
        train = test_lang[i][0]
        test = test_lang[i][1]
        dev = test_lang[i][2]
        emb = test_lang[i][3]
        tagger = test_lang[i][4]
        game2 = initialise_game(train, test, dev, emb, budget, max_seq_len=max_seq_len, max_vocab_size=max_vocab_size)
        robot.update_embeddings(game2.w2v)
        model = CRFTagger(tagger)
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
    parts = all_langs.split(";")
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
    set_logger(args.log_path)
    budget = int(args.budget)
    train_lang = construct_languages(args.train)
    # load the test data: target languages
    test_lang = construct_languages(args.test)
    max_seq_len = args.max_seq_len
    max_vocab_size = args.max_vocab_size
    embedding_size = args.embedding_size
    # play games for training a robot
    robot = play_ner(agent=args.agent, train_lang=train_lang, train_lang_num=len(train_lang), budget=budget,
                     max_seq_len=max_seq_len, max_vocab_size=max_vocab_size, embedding_size=embedding_size,
                     max_episode=args.episode)
    # play a new game with the trained robot
    run_test(robot=robot, test_lang=test_lang, test_lang_num=len(test_lang), budget=budget, max_seq_len=max_seq_len,
             max_vocab_size=max_vocab_size)


if __name__ == '__main__':
    main()
