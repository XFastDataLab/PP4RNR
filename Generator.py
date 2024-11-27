from tensorflow.keras.utils import Sequence
import numpy as np
FLAG_CTR = 1

def fetch_ctr_dim3(News, docids, bucket, flag=1):
    batch_size, doc_num = docids.shape
    doc_imp = News.news_stat_imp[docids]
    doc_click = News.news_stat_click[docids]
    ctr = np.zeros(docids.shape)
    for i in range(batch_size):
        for j in range(doc_num):
            b = bucket[i, j] - 1
            if b < 0:
                b = 0
            ctr[i, j] = doc_click[i, j, b] / (doc_imp[i, j, b] + 0.01)
    ctr = ctr * 200
    ct = np.ceil(ctr)
    ctr = np.array(ct, dtype='int32')
    return ctr


class TrainGenerator(Sequence):
    def __init__(self, News, Users, news_id, user_ids, buckets, news_exist_times, label, uid, batch_size):
        self.News = News
        self.Users = Users

        self.user_ids = user_ids
        self.doc_id = news_id
        self.buckets = buckets
        self.news_exist_times = news_exist_times
        self.label = label
        self.uid = uid

        self.batch_size = batch_size
        self.ImpNum = self.label.shape[0]

    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        ed = (idx + 1) * self.batch_size
        if ed > self.ImpNum:
            ed = self.ImpNum

        doc_ids = self.doc_id[start:ed]
        newsllm = np.expand_dims(doc_ids , axis=-1)

        news_feature = self.News.fetch_news(doc_ids)

        user_ids = self.user_ids[start:ed]
        userid = self.uid[start:ed]

        clicked_ids = self.Users.click[user_ids]

        user_feature = self.News.fetch_news(clicked_ids)

        user_feature_id = userid

        bucket = self.buckets[start:ed]

        news_exist_time = self.news_exist_times[start:ed]

        click_bucket = self.Users.click_bucket[user_ids]
        click_ctr = fetch_ctr_dim3(self.News, clicked_ids, click_bucket, FLAG_CTR)
        label = self.label[start:ed]
        userllm = np.expand_dims(clicked_ids, axis=-1)
        news_feature = np.concatenate((news_feature, newsllm),axis=-1)
        user_feature = np.concatenate((user_feature, userllm),axis=-1)
        return [news_feature, user_feature, bucket, news_exist_time, user_feature_id, click_ctr], [label]


class UserGenerator(Sequence):
    def __init__(self, News, Users, uid, batch_size):
        self.News = News
        self.Users = Users
        self.uid = uid

        self.batch_size = batch_size
        self.ImpNum = self.Users.click.shape[0]

    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        ed = (idx + 1) * self.batch_size
        if ed > self.ImpNum:
            ed = self.ImpNum

        uid = self.uid[start:ed]
        clicked_ids = self.Users.click[start:ed]
        userllm = np.expand_dims(clicked_ids, axis=-1)
        user_feature = self.News.fetch_news(clicked_ids)
        user_feature_id = uid
        click_bucket = self.Users.click_bucket[start:ed]
        user_feature = np.concatenate((user_feature, userllm), axis=-1)
        click_ctr = fetch_ctr_dim3(self.News, clicked_ids, click_bucket, FLAG_CTR)
        return [user_feature, click_ctr]


class PopularityGenerator(Sequence):
    def __init__(self, Impressions, batch_size):
        self.Impressions = Impressions
        self.batch_size = batch_size
        self.ImpNum = len(self.Impressions)

    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        if start > self.ImpNum:
            start = self.ImpNum

        bucket = self.Impressions[start]['tsp']
        bucket = np.array(bucket)

        news_exist_time = self.Impressions[start]['news_exist_time']
        news_exist_time = np.array(news_exist_time)
        news_exist_time = news_exist_time.astype(np.float32)

        return [bucket, news_exist_time]


class PopularityBiasGenerator(Sequence):
    def __init__(self, uid, batch_size):
        self.uid = uid
        self.batch_size = batch_size
        self.ImpNum = len(self.uid)

    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        ed = (idx + 1) * self.batch_size
        if start > self.ImpNum:
            start = self.ImpNum

        uid = self.uid[start:ed]
        return uid


class NewsGenerator(Sequence):
    def __init__(self, News, batch_size):
        self.News = News

        self.batch_size = batch_size
        self.ImpNum = self.News.title.shape[0]

    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        ed = (idx + 1) * self.batch_size
        if ed > self.ImpNum:
            ed = self.ImpNum
        doc_ids = np.array([i for i in range(start, ed)])
        newsllm = np.expand_dims(doc_ids, axis=-1)
        news_feature = self.News.fetch_news(doc_ids)
        news_feature = np.concatenate((news_feature, newsllm), axis=-1)
        return news_feature
