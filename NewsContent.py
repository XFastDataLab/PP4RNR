import pickle

import math

from utils import *
import numpy as np
import os
import json
import tensorflow as tf
from DataGeneration import *
from scipy import stats


class NewsContent:
    def __init__(self, config, ):
        self.news_stat_imp = None
        self.newsid=None
        self.news_stat_click = None
        self.news_publish_bucket2 = None
        self.news = None
        self.news_index = None
        self.category_dict = None
        self.subcategory_dict = None
        self.word_dict = None
        self.title = None
        self.vert = None
        self.subvert = None
        self.body = None
        self.entity_embedding = None
        self.news_entity_index = None
        self.news_entity = None
        self.retain_entities = None
        self.entity_popularity = None
        self.word_popularity = None
        self.news_popularity = None
        self.config = config
        self.dg = DataGeneration(config)
        self.read_news()
        self.get_doc_input()
        self.load_entity()
        self.load_ctr()
        self.load_publish_time()

        if os.path.exists(os.path.join(self.config['popularity_path'], 'entity_popularity.npy')):
            self.entity_popularity = np.load(os.path.join(self.config['popularity_path'], 'entity_popularity.npy'))
        else:
            self.get_entity_popularity(self.config)
        if os.path.exists(os.path.join(self.config['popularity_path'], 'word_popularity.npy')):
            self.word_popularity = np.load(os.path.join(self.config['popularity_path'], 'word_popularity.npy'))
        else:
            self.get_word_popularity(self.config)


    def getllm(self,id):
        vec=[]
        original_dim = id.ndim
        if id.ndim == 1:
            id = id[np.newaxis, :]
        for row in id:
            rowvec=[]
            for elem in row:
                key = int(elem)
                if key in self.newsid:
                    j=self.newsid[key]
                    embedding = self.llm[j]
                else:
                    embedding=np.zeros(768, dtype=np.float32)
                rowvec.append(embedding)
            vec.append(np.stack(rowvec, axis=0))
        vec = np.stack(vec, axis=0)
        vec=vec.astype(np.float32)
        if original_dim == 1:
            vec = vec.squeeze(axis=0)
        return vec
    def fetch_news(self, doc_ids ):
        title = None
        vert = None
        subvert = None
        body = None
        entity = None
        config = self.config
        if 'title' in config['attrs']:
            title = self.title[doc_ids]
        if 'vert' in config['attrs']:
            vert = self.vert[doc_ids]
            vert = vert.reshape(list(doc_ids.shape) + [1])
        if 'subvert' in config['attrs']:
            subvert = self.subvert[doc_ids]
            subvert = subvert.reshape(list(doc_ids.shape) + [1])
        if 'body' in config['attrs']:
            body = self.body[doc_ids]
        if 'entity' in config['attrs']:
            entity = self.news_entity_index[doc_ids]

        FeatureTable = {'title': title, 'vert': vert, 'subvert': subvert, 'body': body, 'entity': entity}
        feature = [FeatureTable[v] for v in config['attrs']]
        feature = np.concatenate(feature, axis=-1)
        return feature

    def read_news(self):
        config = self.config
        news = {}
        category = []
        subcategory = []
        news_index = {}
        index = 1
        word_dict = {}
        newsid={}
        with open(self.config['data_root_path'] + '/news.tsv') as f:
            lines = f.readlines()
        for line in lines:
            split = line.strip('\n').split('\t')
            doc_id, vert, subvert, title = split[0:4]
            if len(split) > 4:
                body = split[-1]
            else:
                body = ''

            if doc_id in news_index:
                continue
            news_index[doc_id] = index
            newsid[index]=doc_id
            index += 1
            category.append(vert)
            subcategory.append(subvert)
            title = title.lower()
            title = word_tokenize(title)
            for word in title:
                if not (word in word_dict):
                    word_dict[word] = 0
                word_dict[word] += 1

            if 'body' in config['attrs']:
                body = body.lower()
                body = word_tokenize(body)
                for word in body:
                    if not (word in word_dict):
                        word_dict[word] = 0
                    word_dict[word] += 1
                news[doc_id] = [vert, subvert, title, body]
            else:
                news[doc_id] = [vert, subvert, title, None]

        category = list(set(category))
        subcategory = list(set(subcategory))
        category_dict = {}
        index = 0
        for c in category:
            category_dict[c] = index
            index += 1
        subcategory_dict = {}
        index = 0
        for c in subcategory:
            subcategory_dict[c] = index
            index += 1
        word_dict_true = {}
        word_index = 1
        for word in word_dict:
            if word_dict[word] < config['word_filter']:
                continue
            if not word in word_dict_true:
                word_dict_true[word] = word_index
                word_index += 1


        self.news = news
        self.news_index = news_index
        self.category_dict = category_dict
        self.subcategory_dict = subcategory_dict
        self.word_dict = word_dict_true
        self.newsid=newsid

    def get_doc_input(self):
        config = self.config
        news = self.news
        news_index = self.news_index
        category = self.category_dict
        subcategory = self.subcategory_dict
        word_dict = self.word_dict

        title_length = config['title_length']
        body_length = config['body_length']

        news_num = len(news) + 1
        news_title = np.zeros((news_num, title_length), dtype='int32')
        news_vert = np.zeros((news_num,), dtype='int32')
        news_subvert = np.zeros((news_num,), dtype='int32')
        if 'body' in config['attrs']:
            news_body = np.zeros((news_num, body_length), dtype='int32')
        else:
            news_body = None
        for key in news:
            vert, subvert, title, body = news[key]
            doc_index = news_index[key]
            news_vert[doc_index] = category[vert]
            news_subvert[doc_index] = subcategory[subvert]
            for word_id in range(min(title_length, len(title))):
                if title[word_id].lower() in word_dict:
                    word_index = word_dict[title[word_id].lower()]
                else:
                    word_index = 0
                news_title[doc_index, word_id] = word_index
            if 'body' in config['attrs']:
                for word_id in range(min(body_length, len(body))):
                    word = body[word_id].lower()
                    if word in word_dict:
                        word_index = word_dict[word]
                    else:
                        word_index = 0
                    news_body[doc_index, word_id] = word_index

        self.title = news_title
        self.vert = news_vert
        self.subvert = news_subvert
        self.body = news_body

    def load_entity(self):
        config = self.config
        news_index = self.news_index
        max_entity_num = config['max_entity_num']
        KG_root_path = config['KG_root_path']
        EntityId2Index = self.dg.entity2index
        news_entity = {}
        retain_entities = {}
        index = 1
        g = []
        if os.path.exists(os.path.join(KG_root_path, 'train_news.tsv')):
            with open(os.path.join(KG_root_path, 'train_news.tsv')) as f:
                lines = f.readlines()
            for line in lines:
                split = line.strip('\n').split('\t')
                doc_id, _, _, _, _, _, entities = split[0:7]
                if not doc_id in news_index:
                    continue
                news_entity[doc_id] = []
                entities = json.loads(entities)
                for j in range(len(entities)):
                    e = entities[j]['Label']
                    eid = entities[j]['WikidataId']
                    if not eid in EntityId2Index:
                        continue
                    if not eid in retain_entities:
                        retain_entities[eid] = index
                        index += 1
                    news_entity[doc_id].append([e, eid])
        if os.path.exists(os.path.join(KG_root_path, 'test_news.tsv')):
            with open(os.path.join(KG_root_path, 'test_news.tsv')) as f:
                lines = f.readlines()
            for line in lines:
                split = line.strip('\n').split('\t')
                doc_id, _, _, _, _, _, entities = split[0:7]
                if not doc_id in news_index:
                    continue
                news_entity[doc_id] = []
                entities = json.loads(entities)
                for j in range(len(entities)):
                    e = entities[j]['Label']
                    eid = entities[j]['WikidataId']
                    if not eid in EntityId2Index:
                        continue
                    if not eid in retain_entities:
                        retain_entities[eid] = index
                        index += 1
                    news_entity[doc_id].append([e, eid])
        if os.path.exists(os.path.join(KG_root_path, 'val_news.tsv')):
            with open(os.path.join(KG_root_path, 'val_news.tsv')) as f:
                lines = f.readlines()
            for line in lines:
                split = line.strip('\n').split('\t')
                doc_id, _, _, _, _, _, entities = split[0:7]
                if not doc_id in news_index:
                    continue
                news_entity[doc_id] = []
                entities = json.loads(entities)
                for j in range(len(entities)):
                    e = entities[j]['Label']
                    eid = entities[j]['WikidataId']
                    if not eid in EntityId2Index:
                        continue
                    if not eid in retain_entities:
                        retain_entities[eid] = index
                        index += 1
                    news_entity[doc_id].append([e, eid])

        entity_embedding = np.zeros((len(retain_entities) + 1, 100), dtype='float32')

        temp_entity_embedding = self.dg.entity_values
        for eid in retain_entities:
            retain_index = retain_entities[eid]
            index = EntityId2Index[eid]
            entity_embedding[retain_index, :] = temp_entity_embedding[index]

        news_entity_index = np.zeros((len(news_index) + 1, max_entity_num), dtype='int32')

        for news_id in news_index:
            index = news_index[news_id]
            entities = news_entity[news_id]
            ri = np.random.permutation(len(entities))
            for j in range(min(len(entities), max_entity_num)):
                eid = entities[ri[j]][-1]
                news_entity_index[index, j] = retain_entities[eid]

        self.entity_embedding = entity_embedding
        self.news_entity_index = news_entity_index
        self.news_entity = news_entity
        self.retain_entities = retain_entities

    def get_entity_popularity(self, config):
        day = self.config['day']
        n = math.ceil((504 / day))
        entity_click = np.zeros((n + 1, len(self.retain_entities) + 1), dtype='float32')
        entity_exposure = np.zeros((n + 1, len(self.retain_entities) + 1), dtype='float32')
        with tf.io.gfile.GFile(self.config['data_root_path'] + '/behaviors.tsv', "r") as r:
            for line in r:
                uid, Time, history, impr = line.strip("\n").split('\t')[-4:]
                Time = parse_time_bucket(Time)
                Time = math.ceil(Time / day)
                for j in impr.split():
                    if int(j.split("-")[1]) == 1:
                        entity_click[Time, self.news_entity_index[self.news_index[j.split("-")[0]]]] += 1
                    entity_exposure[Time, self.news_entity_index[self.news_index[j.split("-")[0]]]] += 1
            entity_popularity = entity_click / (entity_exposure + 0.01)
            # entity_popularity = entity_popularity + 0.2
            entity_popularity[:, 0] = 0
            entity_popularity = np.multiply(entity_popularity, 200)
            entity_popularity = np.ceil(entity_popularity)
            entity_popularity = entity_popularity.astype("int32")
            self.entity_popularity = entity_popularity
        np.save(os.path.join(config['popularity_path'], 'entity_popularity.npy'), self.entity_popularity)

    def get_word_popularity(self, config):
        day = self.config['day']
        n = math.ceil((504 / day))
        word_click = np.zeros((n + 1, len(self.word_dict) + 1), dtype='float32')
        word_exposure = np.zeros((n + 1, len(self.word_dict) + 1), dtype='float32')
        with tf.io.gfile.GFile(self.config['data_root_path'] + '/behaviors.tsv', "r") as r:
            for line in r:
                uid, Time, history, impr = line.strip("\n").split('\t')[-4:]
                Time = parse_time_bucket(Time)
                Time = math.ceil(Time / day)
                for j in impr.split():
                    if int(j.split("-")[1]) == 1:
                        word_click[Time, self.title[self.news_index[j.split("-")[0]]]] += 1
                    word_exposure[Time, self.title[self.news_index[j.split("-")[0]]]] += 1
            word_popularity = word_click / (word_exposure + 0.01)
            # word_popularity = word_popularity + 0.2
            word_popularity[:, 0] = 0
            word_popularity = np.multiply(word_popularity, 200)
            word_popularity = np.ceil(word_popularity)
            word_popularity = word_popularity.astype("int32")
            self.word_popularity = word_popularity
        np.save(os.path.join(config['popularity_path'], 'word_popularity.npy'), self.word_popularity)

    def load_ctr(self, ):
        self.news_stat_imp = self.dg.news_stat_imp
        self.news_stat_click = self.dg.news_stat_click

    def load_publish_time(self, ):
        self.news_publish_bucket2 = self.dg.news_publish_time

