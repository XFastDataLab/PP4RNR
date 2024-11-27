from tensorflow.keras import backend as K
from tensorflow.keras.layers import Embedding, concatenate
from tensorflow.keras.layers import Dense, Input, Flatten, average, Lambda
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
import tensorflow as tf
from tensorflow import keras
from tensorflow import einsum
from keras import layers


class AttLayer(layers.Layer):

    def __init__(self, dim=200, seed=0, **kwargs):
        self.W = None
        self.b = None
        self.q = None
        self.dim = dim
        self.seed = seed
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        dim = self.dim
        self.W = self.add_weight(
            name="W",
            shape=(int(input_shape[-1]), dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=(dim,),
            initializer=keras.initializers.Zeros(),
            trainable=True,
        )
        self.q = self.add_weight(
            name="q",
            shape=(dim, 1),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, inputs, mask=None, **kwargs):

        attention = K.tanh(K.dot(inputs, self.W) + self.b)
        attention = K.dot(attention, self.q)

        attention = K.squeeze(attention, axis=2)

        if mask is None:
            attention = K.exp(attention)
        else:
            attention = K.exp(attention) * K.cast(mask, dtype="float32")

        attention_weight = attention / (
            K.sum(attention, axis=-1, keepdims=True) + K.epsilon()
        )

        attention_weight = K.expand_dims(attention_weight)
        weighted_input = inputs * attention_weight
        return K.sum(weighted_input, axis=1)

    def compute_mask(self, input, input_mask=None):
        return None

    def get_config(self):
        config = super(AttLayer, self).get_config()
        config.update({'W': self.W, 'b': self.b, 'q': self.q, 'dim': self.dim,
                       'seed': self.seed})
        return config

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


class Self_Attention(layers.Layer):
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.WQ = None
        self.WK = None
        self.WV = None
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Self_Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len is None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], self.output_dim

    def call(self, x):
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))

        Q_seq_reshape = K.reshape(Q_seq, (-1, K.shape(Q_seq)[2], K.shape(Q_seq)[3]))
        K_seq_reshape = K.reshape(K_seq, (-1, K.shape(K_seq)[2], K.shape(K_seq)[3]))
        A = K.batch_dot(Q_seq_reshape, K_seq_reshape, axes=[2, 2]) / self.size_per_head ** 0.5
        A = K.reshape(A, (-1, K.shape(Q_seq)[1], K.shape(A)[1], K.shape(A)[2]))


        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = K.softmax(A)


        A_reshape = K.reshape(A, (-1, K.shape(A)[2], K.shape(A)[3]))
        V_seq_reshape = K.reshape(V_seq, (-1, K.shape(V_seq)[2], K.shape(V_seq)[3]))
        O_seq = K.batch_dot(A_reshape, V_seq_reshape, axes=[2, 1])
        O_seq = K.reshape(O_seq, (-1, K.shape(A)[1], K.shape(O_seq)[1], K.shape(O_seq)[2]))

        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def get_config(self):
        config = super(Self_Attention, self).get_config()
        config.update({'WQ': self.WQ, 'WK': self.WK, 'WV': self.WV, 'nb_head': self.nb_head,
                       'size_per_head': self.size_per_head, 'output_dim': self.output_dim})
        return config


def AttentivePooling(dim1, dim2):
    vecs_input = Input(shape=(dim1, dim2), dtype='float32')  # (50,400)
    user_vecs = Dropout(0.2)(vecs_input)
    user_att = Dense(200, activation='tanh')(user_vecs)  # (50,200)
    user_att = Flatten()(Dense(1)(user_att))  # (50,)
    user_att = Activation('softmax')(user_att)  # (50,)
    user_vec = keras.layers.Dot((1, 1))([user_vecs, user_att])  # (400,)
    model = Model(vecs_input, user_vec)
    return model


def AttentivePoolingQKY(dim1, dim2, dim3,droppro=0.2):
    vecs_input = Input(shape=(dim1, dim2), dtype='float32')
    value_input = Input(shape=(dim1, dim3), dtype='float32')
    user_vecs = Dropout(droppro)(vecs_input)
    user_att = Dense(200, activation='tanh')(user_vecs)
    user_att = Flatten()(Dense(1)(user_att))
    user_att = Dropout(droppro)(user_att, training=True)
    user_att = Activation('softmax')(user_att)
    user_vec = keras.layers.Dot((1, 1))([value_input, user_att])
    model = Model([vecs_input, value_input], user_vec)
    return model

def AttentivePooling_bias(dim1, dim2, dim3):
    bias_input = Input(shape=(dim1, dim2), dtype='float32')
    value_input = Input(shape=(dim1, dim3), dtype='float32')
    bias_vecs = Dropout(0.2)(bias_input)
    user_att = Dense(200, activation='tanh')(bias_vecs)
    user_att = Flatten()(Dense(1)(user_att))
    user_att = Activation('softmax')(user_att)
    user_vec = keras.layers.Dot((1, 1))([value_input, user_att])
    model = Model([bias_input, value_input], user_vec)
    return model
