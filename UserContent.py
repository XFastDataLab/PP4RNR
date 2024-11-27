import os
import random

import numpy as np

from DataGeneration import *


def new_sample(nn, ratio):
    if ratio > len(nn):
        return random.sample(nn * (ratio // len(nn) + 1), ratio)
    else:
        return random.sample(nn, ratio)

class UserContent:
    def __init__(self, news_index, entity_popularity, news_entity_index, word_popularity, news_title,
                 config, filename):
        self.click_bucket = None
        self.click = None
        self.session = None
        self.news_index = news_index
        self.entity_popularity = entity_popularity
        self.news_entity_index = news_entity_index
        self.word_popularity = word_popularity
        self.news_title = news_title
        self.config = config
        self.filename = filename
        self.dg = DataGeneration(config)
        self.load_session()
        self.parse_user()

    def load_session(self, ):
        config = self.config
        path = config['data_root_path']
        day = config['day']
        with open(os.path.join(path, self.filename)) as f:
            lines = f.readlines()
        session = []
        for i in range(len(lines)):
            uid, Time, history, impr = lines[i].strip().split('\t')[-4:]
            click_ids = [j for j in history.split() if np.any(self.news_entity_index[self.news_index[j]])]
            pos = [j.split("-")[0] for j in impr.split() if int(j.split("-")[1]) == 1
                   and np.any(self.news_entity_index[self.news_index[j.split("-")[0]]])]
            neg = [j.split("-")[0] for j in impr.split() if int(j.split("-")[1]) == 0
                   and np.any(self.news_entity_index[self.news_index[j.split("-")[0]]])]
            news_current_time = parse_time_bucket(Time)
            Time = parse_time_bucket(Time)
            Time = math.ceil(Time / day)
            click_bucket = []
            for j in range(len(click_ids)):
                click_bucket.append(Time)
            pos_entity_bucket = [
                self.entity_popularity[Time, self.news_entity_index[self.news_index[k.split("-")[0]]]]
                for k in impr.split() if int(k.split("-")[1]) == 1
                and np.any(self.news_entity_index[self.news_index[k.split("-")[0]]])]

            neg_entity_bucket = [
                self.entity_popularity[Time, self.news_entity_index[self.news_index[k.split("-")[0]]]]
                for k in impr.split() if int(k.split("-")[1]) == 0
                and np.any(self.news_entity_index[self.news_index[k.split("-")[0]]])]
            pos_word_bucket = [self.word_popularity[Time, self.news_title[self.news_index[k.split("-")[0]]]] for k
                               in impr.split() if int(k.split("-")[1]) == 1
                               and np.any(self.news_entity_index[self.news_index[k.split("-")[0]]])]
            neg_word_bucket = [self.word_popularity[Time, self.news_title[self.news_index[k.split("-")[0]]]] for k
                               in impr.split() if int(k.split("-")[1]) == 0
                               and np.any(self.news_entity_index[self.news_index[k.split("-")[0]]])]
            pos_news_exist_time = [
                (news_current_time - self.dg.news_publish_time[self.news_index[k.split("-")[0]]])
                for k in impr.split() if int(k.split("-")[1]) == 1 and np.any(
                    self.news_entity_index[self.news_index[k.split("-")[0]]])]
            neg_news_exist_time = [
                (news_current_time - self.dg.news_publish_time[self.news_index[k.split("-")[0]]])
                for k in impr.split() if int(k.split("-")[1]) == 0 and np.any(
                    self.news_entity_index[self.news_index[k.split("-")[0]]])]
            uid = self.dg.user_index[uid]
            if len(pos) != 0 and len(neg) != 0:
                session.append(
                    [click_ids, pos_entity_bucket, neg_entity_bucket, pos_word_bucket, neg_word_bucket, pos, neg,
                     pos_news_exist_time, neg_news_exist_time, uid, click_bucket])
        self.session = session

    def parse_user(self, ):
        session = self.session
        config = self.config

        MAX_ALL = config['max_clicked_news']
        news_index = self.news_index

        user_num = len(session)
        self.click = np.zeros((user_num, MAX_ALL), dtype='int32')
        self.click_bucket = np.zeros((user_num, MAX_ALL), dtype='int32')

        for user_id in range(len(session)):
            click_ids, _, _, _, _, _, _, _, _, _, click_bucket = session[user_id]
            clicks = []
            for i in range(len(click_ids)):
                clicks.append(news_index[click_ids[i]])

            if len(clicks) > MAX_ALL:
                clicks = clicks[-MAX_ALL:]
                click_bucket = click_bucket[-MAX_ALL:]
            else:
                clicks = [0] * (MAX_ALL - len(click_ids)) + clicks
                click_bucket = [1] * (MAX_ALL - len(click_bucket)) + click_bucket

            self.click[user_id] = np.array(clicks)
            self.click_bucket[user_id] = np.array(click_bucket)
