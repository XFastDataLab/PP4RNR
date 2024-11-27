import math
import sys
from utils import *
import numpy as np
import os
import json
import tensorflow as tf
from utils import *
import matplotlib.pyplot as plt

news_index = {}
user_index = {}
entity2index = {}
entity_values = []
news_publish_time = np.load('./data/popularity/news_publish_time.npy')
with tf.io.gfile.GFile('./data' + '/news.tsv', "r") as rd:
    for line in rd:
        nid, vert, subvert, title, ab, url, _, _ = line.strip("\n").split('\t')
        if nid in news_index:
            continue
        news_index[nid] = len(news_index) + 1

day = 1
n = math.ceil(504 / day)
news_click = np.zeros((n + 1, len(news_index) + 1))
news_exposure = np.zeros((n + 1, len(news_index) + 1))
news_all_click = np.zeros((n + 1))
with tf.io.gfile.GFile('./data' + '/behaviors.tsv', "r") as re:
    for line in re:
        uid, Time, history, impr = line.strip("\n").split('\t')[-4:]
        Time = parse_time_bucket(Time)
        # Time = Time / 24
        Time = math.ceil(Time / day)
        for j in impr.split():
            if int(j.split("-")[1]) == 1:
                news_click[Time, news_index[j.split("-")[0]]] += 1
                news_exposure[Time, news_index[j.split("-")[0]]] += 1
                news_all_click[Time] += 1
            if int(j.split("-")[1]) == 0:
                news_exposure[Time, news_index[j.split("-")[0]]] += 1
plt.figure(figsize=(8, 5))
plt.scatter(news_click[299], news_exposure[299], marker='o', c='w', edgecolors='g')
plt.scatter(news_click[295], news_exposure[295], marker='o', c='w', edgecolors='r')
plt.scatter(news_click[296], news_exposure[296], marker='o', c='w', edgecolors='#FFA500')
plt.scatter(news_click[297], news_exposure[297], marker='o', c='w', edgecolors='b')
plt.scatter(news_click[298], news_exposure[298], marker='o', c='w', edgecolors='#00FF00')
plt.colorbar()

plt.xlabel("popularity", fontsize=15)
plt.ylabel("exposure", fontsize=15)
plt.tick_params(labelsize=15)
plt.savefig('./Pictures/exposure_and_popularity_Result_7.eps', format='eps', bbox_inches='tight',  dpi=400)
