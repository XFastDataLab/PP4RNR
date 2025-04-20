#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import math
import re
import time
from datetime import datetime
import numpy as np
from sklearn.metrics import roc_auc_score
from Generator import *
import tensorflow as tf
import tensorflow_ranking as tfr
import matplotlib.pyplot as plt
from itertools import chain

FLAG_CTR = 1


def word_tokenize(sent):
    pat = re.compile(r'[\w]+|[.,!?;|]')
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []


def trans2tsp(time_str):
    return int(time.mktime(datetime.strptime(time_str, '%m/%d/%Y %I:%M:%S %p').timetuple()))


anchor = trans2tsp('11/02/2019 11:59:59 PM')


def parse_time_bucket(date):
    tsp = trans2tsp(date)
    tsp = tsp - anchor
    tsp = tsp // 3600
    return tsp


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)



def my_auc(label, score):
  
    label = np.array(label)
    score = np.array(score)

    order = np.argsort(score)
    sorted_labels = label[order]

    num_positive = np.sum(sorted_labels == 1)
    num_negative = len(sorted_labels) - num_positive

    if num_positive == 0 or num_negative == 0:
        return None

    positive_ranks = np.sum(np.where(sorted_labels == 1)[0] + 1)

    auc = (positive_ranks - (num_positive * (num_positive + 1) / 2)) / (num_positive * num_negative)
    return auc

def evaluate_performance(rankings, Impressions):
    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 = []
    for i in range(len(Impressions)):
        labels = Impressions[i]['labels']
        labels = np.array(labels)

        score = rankings[i]

        auc = my_auc(labels, score)
        mrr = mrr_score(labels, score)
        ndcg5 = ndcg_score(labels, score, k=5)
        ndcg10 = ndcg_score(labels, score, k=10)

        AUC.append(auc)
        MRR.append(mrr)
        nDCG5.append(ndcg5)
        nDCG10.append(ndcg10)
    AUC = np.array(AUC)
    MRR = np.array(MRR)
    nDCG5 = np.array(nDCG5)
    nDCG10 = np.array(nDCG10)

    AUC = round(AUC.mean(), 4)
    MRR = round(MRR.mean(), 4)
    nDCG5 = round(nDCG5.mean(), 4)
    nDCG10 = round(nDCG10.mean(), 4)

    return AUC, MRR, nDCG5, nDCG10


def evaluate_exposure(rankings, Impressions, News):
    news_click = News.dg.news_click
    exposure = np.zeros(len(news_click))
    for i in range(len(Impressions)):
        score = rankings[i]
        rank = np.argsort(score)[::-1]
        rank = rank[0:10]
        exposure[rank] += 1
    plt.figure(figsize=(8, 5))
    plt.scatter(news_click, exposure, marker='o', c='w', edgecolors='r')
    plt.xlabel("x", fontsize=15)
    plt.ylabel("y", fontsize=15)
    plt.tick_params(labelsize=15)
    plt.show()


def news_ranking(user_scoring, news_scoring, Impressions, popularity_activity_gater, popularity_bias_score):
    rankings = []
    count = 0
    popularity_bias_score = popularity_bias_score.flatten()
    for i in range(len(Impressions)):
        doc_ids = Impressions[i]['docs']
        doc_ids = np.array(doc_ids)
        uv = user_scoring[i]
        nv = news_scoring[doc_ids]
        rel_score = np.dot(nv, uv)
        popularity_bias_scores = popularity_bias_score[count:(count + len(doc_ids))]
        gate = popularity_activity_gater[i]
        score = gate * rel_score + (1 - gate) * popularity_bias_scores
        rankings.append(score)
        count = count + len(doc_ids)
    return rankings
