from AttentionModel import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, Input, TimeDistributed, Dense, Dot, Activation, Lambda, Add, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Flatten, Softmax
from tensorflow.keras import Model
from tensorflow.keras.losses import KLDivergence, CategoricalCrossentropy
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Layer
import tensorflow_probability as tfp
def BPR_LOSS(y_true, y_pred):
    x = tf.where(tf.equal(y_true, 1), y_pred, y_pred)
    x1, x2 = tf.split(x, 2, 1)
    bpr_loss = -K.mean(tf.math.log(tf.sigmoid(tf.subtract(x1, x2))))
    return bpr_loss

def get_embedding_encoder(config, entity_embedding_layer):
    LengthTable = {'title': config['title_length'],
                   'entity': config['max_entity_num']}
    input_length = 0
    PositionTable = {}
    for v in config['attrs']:
        PositionTable[v] = (input_length, input_length + LengthTable[v])
        input_length += LengthTable[v]
    print(PositionTable)
    news_input = Input((input_length,), dtype='int32')
    entity_input = keras.layers.Lambda(lambda x: x[:, PositionTable['entity'][0]:PositionTable['entity'][1]])(
        news_input)
    entity_emb = entity_embedding_layer(entity_input)
    model = Model(news_input, entity_emb)
    return model

def get_news_encoder(config,word_num, word_embedding_matrix, entity_embedding_layer, LLM_ERA,seed):
    llm_era_embedding = Embedding(LLM_ERA.shape[0], LLM_ERA.shape[1],
                              weights=[LLM_ERA], trainable=True)
    LengthTable = {'title': config['title_length'],
                   'entity': config['max_entity_num']}
    input_length = 0
    PositionTable = {}
    for v in config['attrs']:
        PositionTable[v] = (input_length, input_length + LengthTable[v])
        input_length += LengthTable[v]
    word_embedding_layer = Embedding(word_num + 1, word_embedding_matrix.shape[1], weights=[word_embedding_matrix],
                                     trainable=True)
    news_era_input = Input((input_length+1,), dtype='float32')
    news_input,llm_input=news_era_input[..., :input_length], news_era_input[..., input_length:]

    title_input = keras.layers.Lambda(lambda x: x[:, PositionTable['title'][0]:PositionTable['title'][1]])(news_input)
    entity_input = keras.layers.Lambda(lambda x: x[:, PositionTable['entity'][0]:PositionTable['entity'][1]])(news_input)

    title_emb = word_embedding_layer(title_input)
    # title_emb = Dropout(0.2)(title_emb) (batch_size, config['title_length'], 300)

    entity_emb = entity_embedding_layer(entity_input)
    # entity_emb = Dropout(0.2)(entity_emb) #(batch_size, config['max_entity_num'], 100)

    title_co_emb = Self_Attention(20, 20)([title_emb, entity_emb, entity_emb])#(batch_size, config['title_length'], 400)
    entity_co_emb = Self_Attention(20, 20)([entity_emb, title_emb, title_emb])#(batch_size, config['max_entity_num'], 400)

    title_vecs = Self_Attention(20, 20)([title_emb, title_emb, title_emb])#(batch_size, config['title_length'], 400)
    title_vector = keras.layers.Add()([title_vecs, title_co_emb])#(batch_size, config['title_length'], 400)

    title_vector = Dropout(0.2)(title_vector)
    title_vec = AttLayer(400, seed)(title_vector)#(batch_size, 400)

    entity_vecs = Self_Attention(20, 20)([entity_emb, entity_emb, entity_emb])#(batch_size, config['max_entity_num'], 400)
    entity_vector = keras.layers.Add()([entity_vecs, entity_co_emb])#(batch_size, config['max_entity_num'], 400)

    entity_vector = Dropout(0.2)(entity_vector)
    entity_vec = AttLayer(400, seed)(entity_vector)#(batch_size, 400)

    feature = [title_vec, entity_vec]
    news_vec = keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(feature)#(batch_size, 2, 400)
    news_vec = Dropout(0.2)(news_vec)
    news_vecs = AttLayer(400, seed)(news_vec)  # (batch_size, 400)
    news_vecs = AttLayer(400, seed)(news_vec)#(batch_size, 400)

    #cascaded Attention Network

    news_vecs = tf.expand_dims(news_vecs,axis=1)
    news_vecs=TimeDistributed(Dense(400, activation='tanh'))(news_vecs)
    news_llm_era=llm_era_embedding(llm_input)
    news_llm_era= TimeDistributed(Dense(768, activation='tanh'))(news_llm_era)
    #
    #
    llmera_co_emb = Self_Attention(20, 20)([news_llm_era, news_vecs, news_vecs])#1 400
    news_co_emb = Self_Attention(20, 20)([news_vecs, news_llm_era, news_llm_era])#1 400
    llmera_vecs = Self_Attention(20, 20)([news_llm_era, news_llm_era, news_llm_era])#1 400
    llmera_vector = keras.layers.Add()([llmera_vecs, llmera_co_emb])
    llmera_vector = Dropout(0.2)(llmera_vector)
    llmera_vec = AttLayer(400, seed)(llmera_vector)#400


    news_llmera_vecs = Self_Attention(20, 20)([news_vecs, news_vecs, news_vecs])#1 400
    news_llmera_vector = keras.layers.Add()([news_llmera_vecs, news_co_emb])#1 400
    news_llmera_vector = Dropout(0.2)(news_llmera_vector)
    news_llmera_vec = AttLayer(400, seed)(news_llmera_vector)#400

    news_vecs = tf.squeeze(news_vecs,axis=1)
    feature1 = [news_vecs, llmera_vec]
    news_vec1 = keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(feature1)
    news_vec1 = Dropout(0.2)(news_vec1)#2 400
    news_vecs1 = AttLayer(400, seed)(news_vec1)#400
    news_vecs = keras.layers.Add()([news_vecs, news_vecs1])

    model = Model(news_input,news_vecs)
    return model
def get_popularity_encoder(config, seed, t):
    LengthTable = {'title': config['title_length'],
                   'entity': config['max_entity_num']}
    input_length = 0
    PositionTable = {}
    for v in config['attrs']:
        PositionTable[v] = (input_length, input_length + LengthTable[v])
        input_length += LengthTable[v]

    entity_popularity_embedding_layer = Embedding(200, 200, trainable=True)
    time_embedding_layer = Embedding(505, 100, trainable=True)

    all_input = Input((input_length + 1,), dtype='float32')
    news_input = keras.layers.Lambda(lambda x: x[:, :input_length])(all_input)
    news_time_input = keras.layers.Lambda(lambda x: x[:, input_length:input_length + 1])(all_input)

    title_input = keras.layers.Lambda(lambda x: x[:, PositionTable['title'][0]:PositionTable['title'][1]])(news_input)
    entity_input = keras.layers.Lambda(lambda x: x[:, PositionTable['entity'][0]:PositionTable['entity'][1]])(
        news_input)

    title_emb = entity_popularity_embedding_layer(title_input)
    title_emb = Dropout(0.2)(title_emb)

    entity_emb = entity_popularity_embedding_layer(entity_input)
    entity_emb = Dropout(0.2)(entity_emb)

    title_co_emb = Self_Attention(400, 1)([title_emb, entity_emb, entity_emb])
    entity_co_emb = Self_Attention(400, 1)([entity_emb, title_emb, title_emb])

    title_vecs = Self_Attention(400, 1)([title_emb, title_emb, title_emb])
    title_vector = keras.layers.Add()([title_vecs, title_co_emb])

    title_vector = Dropout(0.2)(title_vector)
    title_vec = AttLayer(400, seed)(title_vector)

    entity_vecs = Self_Attention(400, 1)([entity_emb, entity_emb, entity_emb])
    entity_vector = keras.layers.Add()([entity_vecs, entity_co_emb])

    entity_vector = Dropout(0.2)(entity_vector)
    entity_vec = AttLayer(400, seed)(entity_vector)

    feature = [title_vec, entity_vec]
    news_vec = keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(feature)
    news_vec = Dropout(0.2)(news_vec)
    news_vecs = AttLayer(400, seed)(news_vec)

    vec1 = Dense(256, activation='tanh')(news_vecs)
    vec1 = Dense(256)(vec1)
    vec1 = Dense(128, )(vec1)
    popularity_score = Dense(1, activation='sigmoid')(vec1)
    time_emb = time_embedding_layer(news_time_input)
    vec2 = Dense(64, activation='tanh')(time_emb)
    vec2 = Dense(64)(vec2)
    popularity_recency_score = Dense(1, activation='sigmoid')(vec2)
    popularity_recency_score = tf.keras.layers.Reshape((1,))(popularity_recency_score)

    popularity_fusion_score = keras.layers.Lambda(lambda x: x[0] * ((1/x[1])**t))([popularity_score, popularity_recency_score])

    model = Model(all_input, popularity_fusion_score)
    return model


def popularity_fusion(config):
    LengthTable = {'title': config['title_length'],
                   'entity': config['max_entity_num']}
    input_length = 0
    PositionTable = {}
    for v in config['attrs']:
        PositionTable[v] = (input_length, input_length + LengthTable[v])
        input_length += LengthTable[v]
    print(PositionTable)
    news_input_1 = Input((input_length,), dtype='int32')
    news_input_2 = Input((1,), dtype='int32')
    popularity_vec = keras.layers.Concatenate(axis=-1)([news_input_1, news_input_2])
    model = Model([news_input_1, news_input_2], popularity_vec)
    return model


def infoNCE_loss(user_vec, user_vec1, temperature=0.1):
    similarity_matrix = tf.matmul(user_vec, user_vec1, transpose_b=True) / temperature
    batch_size = tf.shape(user_vec)[0]
    labels = tf.range(batch_size)
    labels_one_hot = tf.one_hot(labels, depth=batch_size)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=similarity_matrix)
    loss = tf.reduce_mean(loss)
    return loss
def js_divergence_loss(x):
    shape = tf.shape(x)
    gaussian_tensor = tf.random.normal(shape, mean=0.0, stddev=1.0)
    x_prob = tfp.distributions.Categorical(logits=x)
    gaussian_prob = tfp.distributions.Categorical(logits=gaussian_tensor)
    kl_div1 = tfp.distributions.kl_divergence(x_prob, gaussian_prob)
    kl_div2 = tfp.distributions.kl_divergence(gaussian_prob, x_prob)
    js_divergence = 0.5 * (kl_div1 + kl_div2)
    loss = tf.reduce_mean(js_divergence)
    return loss
def combined_loss(y_true, y_pred, user_vecs,perturbed_user_vec,perturbation_vec,candidate_vecs,perturbed_candidate_vec,candidate_perturbation_vec):
    alpha1, alpha2, alpha3 = 0.6, 0.3, 0.1
    #alpha3=1-alpha1-alpha2
    bpr=BPR_LOSS(y_true, y_pred)
    #bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    user_infoNCEloss = infoNCE_loss(user_vecs, perturbed_user_vec,temperature=0.1)
    candidate_infoNCEloss = infoNCE_loss(candidate_vecs, perturbed_candidate_vec, temperature=0.1)
    kl=js_divergence_loss(perturbation_vec)+js_divergence_loss(candidate_perturbation_vec)
    return alpha1*bpr+alpha2*(user_infoNCEloss+candidate_infoNCEloss)+alpha3*kl

class BERTPositionEmbedding(Layer):
    def __init__(self, max_position, embedding_dim, **kwargs):
        super(BERTPositionEmbedding, self).__init__(**kwargs)
        self.max_position = max_position
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        self.position_embeddings = self.add_weight(
            name="position_embeddings",
            shape=(self.max_position, self.embedding_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        super(BERTPositionEmbedding, self).build(input_shape)

    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]
        position_ids = tf.range(seq_length, dtype=tf.int32)
        position_embeddings = tf.gather(self.position_embeddings, position_ids)
        batch_size = tf.shape(inputs)[0]
        position_embeddings = tf.expand_dims(position_embeddings, axis=0)
        position_embeddings = tf.tile(position_embeddings, [batch_size, 1, 1])

        return position_embeddings
def create_pe_model(config, model_config, News, word_embedding_matrix, entity_embedding_matrix, LLM_ERA,seed):
    max_clicked_news = config['max_clicked_news']
    t = model_config['popularity_time']
    dim=400
    seqlen = 50
    entity_embedding_layer = Embedding(entity_embedding_matrix.shape[0], entity_embedding_matrix.shape[1],
                                       weights=[entity_embedding_matrix], trainable=True)
    user_embedding_layer = Embedding(len(News.dg.user_index) + 1, 400, trainable=True)

    popularity_encoder = get_popularity_encoder(config, seed, t)
    news_encoder = get_news_encoder(config, len(News.word_dict), word_embedding_matrix, entity_embedding_layer, LLM_ERA,seed)

    popularity_fusion_encoder = popularity_fusion(config)

    news_input_length = int(news_encoder.input.shape[1])


    clicked_input = Input(shape=(max_clicked_news, news_input_length), dtype='int32')  
    uid = Input(shape=(1, ), dtype='int32')
    candidates = keras.Input((1 + config['np_ratio'], news_input_length), dtype='int32')  
    clicked_buckets = keras.Input((1 + config['np_ratio'], 35), dtype='float32')
    news_time = keras.Input((1 + config['np_ratio'], ), dtype='float32')
    click_ctr = Input(shape=(max_clicked_news,), dtype='int32')

    reshaped_input = tf.reshape(clicked_input, [-1, news_input_length])
    user_vecs111 = news_encoder(reshaped_input) 
    user_vecs = tf.reshape(user_vecs111, [-1, max_clicked_news, dim])
    uid_vecs = tf.keras.layers.Reshape((400,))(user_embedding_layer(uid))


    if config['user_encoder_name'] == 'popularity_user_modeling':
        popularity_embedding_layer = Embedding(200, dim, trainable=True)
        popularity_embedding = popularity_embedding_layer(click_ctr)
        position_embedding_layer=BERTPositionEmbedding(seqlen, dim)
        position_embeddings = position_embedding_layer(user_vecs)

        popularity_dense = TimeDistributed(Dense(units=dim, activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l2(0.01)))(popularity_embedding)
        position_embeddings = TimeDistributed(Dense(units=dim, activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l2(0.01)))(position_embeddings)

        concatenated = Concatenate(axis=-1)([position_embeddings, popularity_dense])

        perturbation_vec = TimeDistributed(Dense(units=dim, activation="tanh"))(concatenated)
        perturbation_vec = TimeDistributed(LayerNormalization())(perturbation_vec)

        perturbed_user_vec= user_vecs +perturbation_vec
        MHSA = Self_Attention(20, 20)
        user_vecs = MHSA([user_vecs, user_vecs, user_vecs])
        perturbed_user_vec= MHSA([perturbed_user_vec, perturbed_user_vec, perturbed_user_vec])

        user_vec_query = keras.layers.Concatenate(axis=-1)([user_vecs, popularity_embedding])
        perturbed_user_vec_query = keras.layers.Concatenate(axis=-1)([perturbed_user_vec,popularity_embedding])

        user_vec = AttentivePoolingQKY(50, 2*dim, dim,0.2)([user_vec_query, user_vecs])
        perturbed_user_vec= AttentivePoolingQKY(50, 2*dim, dim,0.1)([perturbed_user_vec_query, user_vecs])
    else:
        user_vecs = Self_Attention(20, 20)([user_vecs, user_vecs, user_vecs])
        user_vecs = Dropout(0.2)(user_vecs)
        user_vec = AttentivePooling(max_clicked_news, 400)(user_vecs)

    candidate_vecs = news_encoder(candidates)
    rel_scores = keras.layers.Dot(axes=-1)([user_vec, candidate_vecs])
    news_times = tf.keras.layers.Reshape((1 + config['np_ratio'], 1,))(news_time)
    popularity_vec = keras.layers.Concatenate(axis=-1)([clicked_buckets, news_times])

    position_embedding_layer = BERTPositionEmbedding(seqlen, dim)
    position_embeddings = position_embedding_layer(candidate_vecs)
    popularity_dense = TimeDistributed(Dense(units=dim, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01)))(popularity_vec)
    position_embeddings = TimeDistributed(Dense(units=dim, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01)))(position_embeddings)
    concatenated = Concatenate(axis=-1)([position_embeddings, popularity_dense])
    candidate_perturbation_vec = TimeDistributed(Dense(units=dim, activation="tanh"))(concatenated)
    candidate_perturbation_vec = TimeDistributed(LayerNormalization())(candidate_perturbation_vec)
    perturbed_candidate_vec = candidate_vecs + candidate_perturbation_vec

    popularity_score = TimeDistributed(popularity_encoder)(popularity_vec)
    popularity_scores = tf.keras.layers.Reshape((1 + config['np_ratio'],))(popularity_score)

    user_vec_input = keras.layers.Input((dim,), )
    activity_gate = Dense(128, activation='tanh')(user_vec_input)
    activity_gate = Dense(64)(activity_gate)
    activity_gate = Dense(1, activation='sigmoid')(activity_gate)
    activity_gate = keras.layers.Reshape((1,))(activity_gate)
    activity_gater = Model(user_vec_input, activity_gate)
    user_activtiy = activity_gater(user_vec)

    scores = []
    rel_scores = keras.layers.Lambda(lambda x: 2 * x[0] * x[1])([rel_scores, user_activtiy])
    scores.append(rel_scores)
    bias_score = keras.layers.Lambda(lambda x: 2 * x[0] * (1 - x[1]))([popularity_scores, user_activtiy])
    scores.append(bias_score)
    scores = keras.layers.Add()(scores)

    logits = keras.layers.Activation(keras.activations.softmax, name='recommend')(scores)
    model = Model([candidates, clicked_input, clicked_buckets, news_time, uid, click_ctr], [logits])
    model.compile(loss=lambda y_true, y_pred: combined_loss(y_true, y_pred, user_vec, perturbed_user_vec,perturbation_vec,candidate_vecs,perturbed_candidate_vec,candidate_perturbation_vec),
                  optimizer=optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                  metrics=[tf.keras.metrics.Accuracy()])

    user_encoder = Model([clicked_input, click_ctr], user_vec)
    return model, user_encoder, news_encoder, activity_gater, popularity_encoder, popularity_fusion_encoder





