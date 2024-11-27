import math
import sys
from utils import *
import numpy as np
import os
import json
import tensorflow as tf
from utils import *


class DataGeneration:
    def __init__(self, config):
        self.news_stat_click = None
        self.news_stat_imp = None
        self.news_publish_time = None
        self.news_index = {}
        self.user_index = {}
        self.config = config
        self.entity2index = {}
        self.entity_values = []
        self.filename = config['KG_root_path']
        with tf.io.gfile.GFile(self.config['data_root_path'] + '/news.tsv', "r") as rd:
            for line in rd:
                nid, vert, subvert, title, ab, url, _, _ = line.strip("\n").split('\t')
                if nid in self.news_index:
                    continue
                self.news_index[nid] = len(self.news_index) + 1
        self.load_user_index()
        self.load_entity2index()
        if os.path.exists(os.path.join(self.config['popularity_path'], 'news_publish_time.npy')):
            self.news_publish_time = np.load(os.path.join(self.config['popularity_path'], 'news_publish_time.npy'))
            print(self.news_publish_time[self.news_index['N36786']])
        else:
            self.load_news_publish_time()
        if os.path.exists(os.path.join(self.config['popularity_path'], 'news_stat_click.npy')):
            self.news_stat_click = np.load(os.path.join(self.config['popularity_path'], 'news_stat_click.npy'))
        else:
            self.load_news_clicks()
        if os.path.exists(os.path.join(self.config['popularity_path'], 'news_stat_imp.npy')):
            self.news_stat_imp = np.load(os.path.join(self.config['popularity_path'], 'news_stat_imp.npy'))
        else:
            self.load_news_exposure()

    def load_entity2index(self):
        index = 0
        if os.path.exists(os.path.join(self.filename, 'train_entity_embedding.vec')):
            with open(os.path.join(self.filename, 'train_entity_embedding.vec'), 'r',
                      newline='') as fr1:
                for line in fr1:
                    e = line.split()
                    if not e[0] in self.entity2index:
                        self.entity2index[e[0]] = index
                        index += 1
                    self.entity_values.append(e[1:])
        if os.path.exists(os.path.join(self.filename, 'test_entity_embedding.vec')):
            with open(os.path.join(self.filename, 'test_entity_embedding.vec'), 'r',
                      newline='') as fr2:
                for line in fr2:
                    e = line.split()
                    if not e[0] in self.entity2index:
                        self.entity2index[e[0]] = index
                        index += 1
                    self.entity_values.append(e[1:])
        if os.path.exists(os.path.join(self.filename, 'val_entity_embedding.vec')):
            with open(os.path.join(self.filename, 'val_entity_embedding.vec'), 'r',
                      newline='') as fr3:
                for line in fr3:
                    e = line.split()
                    if not e[0] in self.entity2index:
                        self.entity2index[e[0]] = index
                        index += 1
                    self.entity_values.append(e[1:])

    def load_user_index(self):
        config = self.config
        path = config['data_root_path']
        with tf.io.gfile.GFile(path + '/behaviors.tsv', "r") as rd:
            for line in rd:
                uid, Time, history, impr = line.strip("\n").split('\t')[-4:]
                if uid in self.user_index:
                    continue
                self.user_index[uid] = len(self.user_index) + 1

    def load_news_publish_time(self):
        config = self.config
        path = config['data_root_path']
        news_id = np.ones(len(self.news_index) + 1) * np.inf
        with tf.io.gfile.GFile(path + '/behaviors.tsv', "r") as rd:
            for line in rd:
                uid, Time, history, impr = line.strip("\n").split('\t')[-4:]
                for i in impr.split():
                    if parse_time_bucket(Time) <= news_id[self.news_index[i.split("-")[0]]]:
                        news_id[self.news_index[i.split("-")[0]]] = parse_time_bucket(Time)
                for j in history.split():
                    if parse_time_bucket(Time) <= news_id[self.news_index[j]]:
                        news_id[self.news_index[j]] = parse_time_bucket(Time)
        news_id[0] = 0
        self.news_publish_time = news_id
        np.save(os.path.join(config['popularity_path'], 'news_publish_time.npy'), self.news_publish_time)

    def load_news_clicks(self):
        config = self.config
        day = config['day']
        path = config['data_root_path']
        n = math.ceil(504 / day)
        news_click = np.zeros((len(self.news_index) + 1, n + 1))
        with tf.io.gfile.GFile(path + '/behaviors.tsv', "r") as re:
            for line in re:
                uid, Time, history, impr = line.strip("\n").split('\t')[-4:]
                Time = parse_time_bucket(Time)
                Time = math.ceil(Time / day)
                for j in impr.split():
                    if int(j.split("-")[1]) == 1:
                        news_click[self.news_index[j.split("-")[0]], Time] += 1
        self.news_stat_click = news_click
        np.save(os.path.join(config['popularity_path'], 'news_stat_click.npy'), self.news_stat_click)

    def load_news_exposure(self):
        config = self.config
        day = config['day']
        path = config['data_root_path']
        n = math.ceil(504 / day)
        news_exposure = np.zeros((len(self.news_index) + 1, n + 1))
        with tf.io.gfile.GFile(path + '/behaviors.tsv', "r") as re:
            for line in re:
                uid, Time, history, impr = line.strip("\n").split('\t')[-4:]
                Time = parse_time_bucket(Time)
                Time = math.ceil(Time / day)
                for j in impr.split():
                    news_exposure[self.news_index[j.split("-")[0]], Time] += 1
        self.news_stat_imp = news_exposure
        np.save(os.path.join(config['popularity_path'], 'news_stat_imp.npy'), self.news_stat_imp)

