import numpy as np
import random
import os

def news_ample(nn, ratio):
    if ratio > len(nn):
        return random.sample(nn * (ratio // len(nn) + 1), ratio)
    else:
        return random.sample(nn, ratio)

def get_train_input(session, news_index, config):
    np_ratio = config['np_ratio']
    max_entity_num = config['max_entity_num']
    max_title_num = config['title_length']
    sess_pos = []
    sess_neg = []
    user_id = []
    uid = []
    sess_pos_entity_bucket = []
    sess_neg_entity_bucket = []
    sess_pos_word_bucket = []
    sess_neg_word_bucket = []
    sess_pos_news_exist_time = []
    sess_neg_news_exist_time = []
    for sess_id in range(len(session)):
        sess = session[sess_id]
        click_ids, pos_entity_bucket, neg_entity_bucket, pos_word_bucket, neg_word_bucket, poss, negs, pos_news_exist_time, neg_news_exist_time, u_id, click_bucket = sess
        for i in range(len(poss)):
            pos = poss[i]
            neg = news_ample(negs, np_ratio)
            sess_pos.append(pos)
            sess_neg.append(neg)
            sess_pos_entity_bucket.append(pos_entity_bucket[i])
            sess_pos_word_bucket.append(pos_word_bucket[i])
            sess_pos_news_exist_time.append(pos_news_exist_time[i])
            neg = ",".join(neg)
            sess_neg_entity_bucket.append(neg_entity_bucket[negs.index(neg)])
            sess_neg_word_bucket.append(neg_word_bucket[negs.index(neg)])
            sess_neg_news_exist_time.append(neg_news_exist_time[negs.index(neg)])
            user_id.append(sess_id)
            uid.append(u_id)
    sess_all = np.zeros((len(sess_pos), 1 + np_ratio), dtype='int32')
    sess_news_exist_time = np.zeros((len(sess_pos), 1 + np_ratio), dtype='int32')
    sess_buckets = np.zeros((len(sess_pos), 1 + np_ratio, (max_entity_num+max_title_num)), dtype='float32')
    label = np.zeros((len(sess_pos), 1 + np_ratio), dtype='int32')
    for sess_id in range(sess_all.shape[0]):
        pos = sess_pos[sess_id]
        negs = sess_neg[sess_id]
        sess_all[sess_id, 0] = news_index[pos]
        sess_buckets[sess_id, 0, :] = np.concatenate((sess_pos_word_bucket[sess_id], sess_pos_entity_bucket[sess_id]), axis=0)
        sess_news_exist_time[sess_id, 0] = sess_pos_news_exist_time[sess_id]
        index = 1
        for neg in negs:
            sess_all[sess_id, index] = news_index[neg]
            sess_buckets[sess_id, index, :] = np.concatenate((sess_neg_word_bucket[sess_id], sess_neg_entity_bucket[sess_id]), axis=0)
            sess_news_exist_time[sess_id, index] = sess_neg_news_exist_time[sess_id]
            index += 1
        label[sess_id, 0] = 1

    user_id = np.array(user_id, dtype='int32')
    uid = np.array(uid, dtype='int32')
    return sess_all, sess_buckets, sess_news_exist_time, user_id, label, uid


def get_test_input(session, news_index): 
    Impressions = []
    userid = []
    uid = []
    for sess_id in range(len(session)):
        _, pos_entity_bucket, neg_entity_bucket, pos_word_bucket, neg_word_bucket, poss, negs, pos_news_exist_time, neg_news_exist_time, u_id, click_bucket = session[sess_id]
        imp = {'labels': [],
               'docs': [],
               'tsp': [],
               'news_exist_time': []}
        userid.append(sess_id)
        uid.append(u_id)
        for i in range(len(poss)):
            docid = news_index[poss[i]]
            pos_bucket = np.concatenate((pos_word_bucket[i], pos_entity_bucket[i]), axis=0)
            imp['docs'].append(docid)
            imp['labels'].append(1)
            imp['tsp'].append(pos_bucket)
            imp['news_exist_time'].append(pos_news_exist_time[i])
        for i in range(len(negs)):
            docid = news_index[negs[i]]
            neg_bucket = np.concatenate((neg_word_bucket[i], neg_entity_bucket[i]), axis=0)
            imp['docs'].append(docid)
            imp['labels'].append(0)
            imp['tsp'].append(neg_bucket)
            imp['news_exist_time'].append(neg_news_exist_time[i])
        Impressions.append(imp)

    userid = np.array(userid, dtype='int32')
    uid = np.array(uid, dtype='int32')
    return Impressions, userid, uid


def load_matrix(embedding_path, word_dict):
    embedding_matrix = np.zeros((len(word_dict) + 1, 300), dtype='float32')
    have_word = []
    with open(os.path.join(embedding_path, 'glove.840B.300d.txt'), 'rb') as f:
        while True:
            l = f.readline()
            if len(l) == 0:
                break
            l = l.split()
            word = l[0].decode()
            if word in word_dict:
                index = word_dict[word]
                tp = [float(x) for x in l[1:]]
                embedding_matrix[index] = np.array(tp)
                have_word.append(word)
    return embedding_matrix, have_word
