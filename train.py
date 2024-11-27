import keras.models

from NewsContent import *
from UserContent import *
from preprocessing import *
from Generator import *
import Generator
from AttentionModel import *
from utils import *
from model import *

import os
import numpy as np
import json
import random
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))
print(tf.test.is_gpu_available())


seed = 25
os.environ['PYTHONHASHSEED'] = str(seed)
tf.compat.v1.set_random_seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '0'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
batchsize=16

# KG_root_path = None
model_save = ('./model.h5')
config = {
          'title_length': 30,
          'body_length': 100,
          'max_clicked_news': 50,
          'np_ratio': 1,
          'news_encoder_name': "SelfAtt",
          'user_encoder_name': "popularity_user_modeling",
          'attrs': ['title', 'entity'],
          'word_filter': 0,
          'data_root_path': './data',
          'embedding_path': './data/word_embedding',
          'KG_root_path': './data/entity',
          'popularity_path': './data/popularity',
          'max_entity_num': 5,
          'day': 1,
          'popularity_hp': 1,
          'title_entity_popularity': 0.5,
          'combine': "mean"
          }

News = NewsContent(config)
News.maxtrix()
TrainUsers = UserContent(News.news_index, News.entity_popularity, News.news_entity_index, News.word_popularity,
                         News.title, config, 'train.tsv')
ValidUsers = UserContent(News.news_index, News.entity_popularity, News.news_entity_index, News.word_popularity,
                         News.title, config, 'val.tsv')
TestUsers = UserContent(News.news_index, News.entity_popularity, News.news_entity_index, News.word_popularity,
                         News.title, config, 'test.tsv')


train_sess, train_buckets, train_news_exist_time, train_user_id, train_label, uid = get_train_input(TrainUsers.session,
                                                                                                    News.news_index,
                                                                                                    config)
val_impressions, val_userids, val_uid = get_test_input(ValidUsers.session, News.news_index)
test_impressions, test_userids, test_uid= get_test_input(TestUsers.session, News.news_index)


title_word_embedding_matrix, have_word = load_matrix(embedding_path, News.word_dict)

train_generator = TrainGenerator(News, TrainUsers, train_sess, train_user_id, train_buckets, train_news_exist_time,
                                 train_label, uid, batchsize)
val_user_generator = UserGenerator(News, ValidUsers, val_uid, batchsize)
test_user_generator = UserGenerator(News, TestUsers, test_uid, 2)
val_popularity_generator = PopularityGenerator(val_impressions, 1)
test_popularity_generator = PopularityGenerator(test_impressions, 1)
news_generator = NewsGenerator(News, batchsize)
LLM_ERA= np.load('./LLMERA.npy')


for i in range(5):
    model_config = {
        'fusion_hp': 0.2,
        'popularity_hp': 0.2,
        'popularity_time': 2}
    # seed = 25
    print(model_config['popularity_time'])
    model, user_encoder, news_encoder, activity_gater, popularity_encoder, popularity_fusion_encoder = create_pe_model(
        config, model_config, News, title_word_embedding_matrix, News.entity_embedding, LLM_ERA,seed)

#
    if os.path.exists(model_save):
        print('Loading**********************')
        model.load_weights(model_save)

    model.summary
    model.fit(train_generator, epochs=1)
    model.save_weights(model_save)

    news_scoring = news_encoder.predict(news_generator, verbose=True)
    user_scoring = user_encoder.predict_generator(test_user_generator, verbose=True)
    val_user_scoring = user_encoder.predict(val_user_generator, verbose=True)
    val_activity_gater = activity_gater.predict(val_user_scoring, verbose=True)
    test_activity_gater = activity_gater.predict(user_scoring, verbose=True)
    val_popularity_vec = popularity_fusion_encoder.predict(val_popularity_generator, verbose=True)
    test_popularity_vec = popularity_fusion_encoder.predict(test_popularity_generator, verbose=True)
    val_popularity = popularity_encoder.predict(val_popularity_vec, verbose=True)
    test_popularity = popularity_encoder.predict(test_popularity_vec, verbose=True)
    val_activity_gater = val_activity_gater[:, 0]
    test_activity_gater = test_activity_gater[:, 0]

    test_rankings = news_ranking(user_scoring, news_scoring, test_impressions, test_activity_gater, test_popularity)
    val_rankings = news_ranking(val_user_scoring, news_scoring, val_impressions, val_activity_gater, val_popularity)
    AUC_1, MRR_1, nDCG5_1, nDCG10_1 = evaluate_performance(test_rankings, test_impressions)
    print('test  AUC:', AUC_1, 'MRR:', MRR_1, 'nDCG5:', nDCG5_1, 'nDCG10:', nDCG10_1)
    AUC_2, MRR_2, nDCG5_2, nDCG10_2 = evaluate_performance(val_rankings, val_impressions)
    print('AUC:', AUC_2, 'MRR:', MRR_2, 'nDCG5:', nDCG5_2, 'nDCG10:', nDCG10_2)

