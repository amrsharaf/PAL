import random
import helpers
import tagger
import logging


# TODO change this to a Gym environment
# TODO fix the random seed
class NERGame:

    def __init__(self, story, test, dev, max_len, w2v, budget):
        # build environment
        # load data as story
        logging.info('Initializing the game:')
        # import story
        self.train_x, self.train_y, self.train_idx, self.train_idy = story
        self.test_x, self.test_y, self.test_idx, self.test_idy = test
        self.dev_x, self.dev_y, self.dev_idx, self.dev_idy = dev
        self.test_sents = helpers.data2sents(self.dev_x, self.dev_y)
        # TODO this should be a function
        self.X_test = [tagger.sent2features(s) for s in self.test_sents]
        self.Y_true = [tagger.sent2labels(s) for s in self.test_sents]
        self.max_len = max_len
        self.w2v = w2v

        logging.info('Story: length = {}'.format(len(self.train_x)))
        self.order = list(range(0, len(self.train_x)))
        # if re-order, use random.shuffle(self.order)
        # load word embeddings, pretrained - w2v
        # logging.info "Dictionary size", len(self.w2v), "Embedding size",
        # len(self.w2v[0])

        # when queried times is 'budget', then stop
        self.budget = budget
        self.queried_times = 0

        # TODO use ndarrays for everything
        # select pool
        self.queried_set_x = []
        self.queried_set_y = []
        self.queried_set_idx = []
        self.queried_set_idy = []

        # let's start
        self.episode = 0
        # story frame
        self.current_frame = 0
        #self.nextFrame = self.current_frame + 1
        self.terminal = False
        self.make_query = False
        self.performance = 0

    def get_frame(self, model):
        self.make_query = False
        sentence = self.train_x[self.order[self.current_frame]]
        sentence_idx = self.train_idx[self.order[self.current_frame]]
        if model.name == 'CRF':
            confidence = model.get_confidence(sentence)
            predictions = model.get_predictions(sentence)
        else:
            confidence = model.get_confidence(sentence_idx)
            predictions = model.get_predictions(sentence_idx)
        preds_padding = []
        orig_len = len(predictions)
        if orig_len < self.max_len:
            preds_padding.extend(predictions)
            for i in range(self.max_len - orig_len):
                preds_padding.append([0] * 5)
        elif orig_len > self.max_len:
            preds_padding = predictions[0:self.max_len]
        else:
            preds_padding = predictions
        obervation = [sentence_idx, confidence, preds_padding]
        return obervation

    # tagger = crf model
    def feedback(self, action, model):
        reward = 0.
        is_terminal = False
        if action[1] == 1:
            self.make_query = True
            self.query()
            new_performance = self.get_performance(model)
            reward = new_performance - self.performance
            if new_performance != self.performance:
                #reward = 3.
                self.performance = new_performance
            # else:
                #reward = -1.
        else:
            reward = 0.
        # next frame
        if self.queried_times == self.budget:
            self.terminal = True
            is_terminal = True
            # update special reward
            # reward = new_performance * 100
            # prepare the next game
            self.reboot()  # set the current frame = 0
            # TODO update these data together
            next_sentence = self.train_x[self.order[self.current_frame]]
            next_sentence_idx = self.train_idx[self.order[self.current_frame]]
        else:
            self.terminal = False
            next_sentence = self.train_x[self.order[self.current_frame + 1]]
            next_sentence_idx = self.train_idx[self.order[self.current_frame + 1]]
            self.current_frame += 1
        # TODO refactor to remove the if statements
        if model.name == "CRF":
            confidence = model.get_confidence(next_sentence)
            predictions = model.get_predictions(next_sentence)
        else:
            # RNN case
            confidence = model.get_confidence(next_sentence_idx)
            predictions = model.get_predictions(next_sentence_idx)
        preds_padding = []
        orig_len = len(predictions)
        if orig_len < self.max_len:
            preds_padding.extend(predictions)
            for i in range(self.max_len - orig_len):
                preds_padding.append([0] * 5)
        elif orig_len > self.max_len:
            preds_padding = predictions[0:self.max_len]
        else:
            preds_padding = predictions
        next_observation = [next_sentence_idx, confidence, preds_padding]
        return reward, next_observation, is_terminal

    def query(self):
        if self.make_query == True:
            sentence = self.train_x[self.order[self.current_frame]]
            # simulate: obtain the labels
            labels = self.train_y[self.order[self.current_frame]]
            self.queried_times += 1
            # logging.info "Select:", sentence, labels
            self.queried_set_x.append(sentence)
            self.queried_set_y.append(labels)
            self.queried_set_idx.append(self.train_idx[self.order[self.current_frame]])
            self.queried_set_idy.append(self.train_idy[self.order[self.current_frame]])
            logging.info('> Queried times {}'.format(len(self.queried_set_x)))

    def get_performance(self, tagger):
        # train with {queried_set_x, queried_set_y}
        # train with examples: self.model.train(self.queried_set_x, self.queried_set_y)
        if tagger.name == "RNN":
            tagger.train(self.queried_set_idx, self.queried_set_idy)
            performance = tagger.test(self.dev_idx, self.dev_y)
            return performance
        else:
            # CRF case
            train_sents = helpers.data2sents(self.queried_set_x, self.queried_set_y)
            tagger.train(train_sents)
            # test on development data
            performance = tagger.test(self.X_test, self.Y_true)
            # performance = self.model.test2conlleval(self.dev_x, self.dev_y)
            return performance

    def reboot(self):
        # resort story
        # why not use docvecs? TypeError: 'DocvecsArray' object does not
        # support item assignment
        random.shuffle(self.order)
        self.queried_times = 0
        self.terminal = False
        self.queried_set_x = []
        self.queried_set_y = []
        self.queried_set_idx = []
        self.queried_set_idy = []
        self.current_frame = 0
        self.episode += 1
        logging.info('> Next episode {}'.format(self.episode))
