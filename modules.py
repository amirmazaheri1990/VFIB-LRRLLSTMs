from __future__ import print_function
from keras.layers import Input, Embedding, LSTM, Dense, merge, Activation
from keras.models import Model, model_from_json
import keras
import numpy as np
import pickle
import os, glob
import theano.sandbox.cuda
theano.sandbox.cuda.use("gpu0")
import time
import sys, random



def get_textEncoder_newIdea(output_size, l2):
    inputs_left = Input(shape=(10,), name='wordsLeft')
    inputs_right = Input(shape=(10,), name='wordsRight')
    vis_rep_len = 4096
    vis_rep_one_len = 200

    emb = keras.layers.embeddings.Embedding(21106, 500, name='Embdding', W_regularizer=l2)
    emb2 = keras.layers.embeddings.Embedding(21106, 500, name='Embdding2', W_regularizer=l2)
    inputs_left_emb = emb(inputs_left)
    inputs_right_emb = emb(inputs_right)
    inputs_left_emb2 = emb2(inputs_left)
    inputs_right_emb2 = emb2(inputs_right)

    inputs_left2 = keras.layers.BatchNormalization( name='normalizationLEFTWord')(inputs_left_emb)
    inputs_right2 = keras.layers.BatchNormalization( name='normalizationRIGHTWord')(inputs_right_emb)

    inputs_left22 = keras.layers.BatchNormalization( name='normalizationLEFTWord2')(inputs_left_emb2)
    inputs_right22 = keras.layers.BatchNormalization( name='normalizationRIGHTWord2')(inputs_right_emb2)

    lstm_left = LSTM(1500, return_sequences=False, name='leftLSTM', W_regularizer=l2)(inputs_left2)
    lstm_right = LSTM(1500, return_sequences=False, go_backwards=True, name='rightLSTM', W_regularizer=l2)(inputs_right2)


    ##### memmory

    lstm_right_tmp = Dense(500, W_regularizer=l2)(lstm_right)
    lstm_left_tmp = Dense(500, W_regularizer=l2)(lstm_left)

    lstm_right_tmp = keras.layers.BatchNormalization()(lstm_right_tmp)
    lstm_left_tmp = keras.layers.BatchNormalization()(lstm_left_tmp)

    lstm_right_tmp = Activation('relu')(lstm_right_tmp)
    lstm_left_tmp = Activation('relu')(lstm_left_tmp)

    lstm_right_tmp = keras.layers.Reshape((1, 500))(lstm_right_tmp)
    lstm_left_tmp = keras.layers.Reshape((1, 500))(lstm_left_tmp)

    ##### try lstm left/right instead of mem
    inputs_left3 = merge([lstm_right_tmp, inputs_left22, lstm_right_tmp], mode='concat', concat_axis=1)
    inputs_right3 = merge([lstm_left_tmp, inputs_right22, lstm_left_tmp], mode='concat', concat_axis=1)

    lstm_left2 = LSTM(1500, return_sequences=False, name='leftLSTM2', W_regularizer=l2)(inputs_left3)
    lstm_right2 = LSTM(1500, return_sequences=False, go_backwards=True, name='rightLSTM2', W_regularizer=l2)(inputs_right3)

    lstm_out = merge([lstm_left2, lstm_right2, lstm_right, lstm_left], mode='concat', concat_axis=1)




    lstm_out = Dense(output_size, W_regularizer=l2)(lstm_out)
    lstm_out = keras.layers.BatchNormalization()(lstm_out)
    lstm_out = Activation('tanh')(lstm_out)


    model = Model(input=[inputs_left, inputs_right], output=[lstm_out])
    return model


def get_textEncoder_newIdea_p(output_size, l2):
    inputs_left = Input(shape=(10,), name='wordsLeft')
    inputs_right = Input(shape=(10,), name='wordsRight')
    inputs_left_p = Input(shape=(10,), name='wordsLeftP')
    inputs_right_p = Input(shape=(10,), name='wordsRightP')
    vis_rep_len = 4096
    vis_rep_one_len = 200

    emb = keras.layers.embeddings.Embedding(21106, 500, name='Embdding', W_regularizer=l2)
    emb2 = keras.layers.embeddings.Embedding(21106, 500, name='Embdding2', W_regularizer=l2)

    emb_p = keras.layers.embeddings.Embedding(32, 500, name='Embddingp', W_regularizer=l2)
    inputs_left_emb_p = emb_p(inputs_left_p)
    inputs_right_emb_p = emb_p(inputs_right_p)

    inputs_left2_p = keras.layers.BatchNormalization(name='normalizationLEFTWordp')(inputs_left_emb_p)
    inputs_right2_p = keras.layers.BatchNormalization(name='normalizationRIGHTWordp')(inputs_right_emb_p)





    inputs_left_emb = emb(inputs_left)
    inputs_right_emb = emb(inputs_right)
    inputs_left_emb2 = emb2(inputs_left)
    inputs_right_emb2 = emb2(inputs_right)

    inputs_left2 = keras.layers.BatchNormalization( name='normalizationLEFTWord')(inputs_left_emb)
    inputs_right2 = keras.layers.BatchNormalization( name='normalizationRIGHTWord')(inputs_right_emb)

    inputs_left22 = keras.layers.BatchNormalization( name='normalizationLEFTWord2')(inputs_left_emb2)
    inputs_right22 = keras.layers.BatchNormalization( name='normalizationRIGHTWord2')(inputs_right_emb2)


    inputs_left2 = merge([inputs_left2,inputs_left2_p])
    inputs_left22 = merge([inputs_left22, inputs_left2_p])

    inputs_right2 = merge([inputs_right2, inputs_right2_p])
    inputs_right22 = merge([inputs_right22, inputs_right2_p])



    lstm_left = LSTM(1500, return_sequences=False, name='leftLSTM', W_regularizer=l2)(inputs_left2)
    lstm_right = LSTM(1500, return_sequences=False, go_backwards=True, name='rightLSTM', W_regularizer=l2)(inputs_right2)


    ##### memmory

    lstm_right_tmp = Dense(500, W_regularizer=l2)(lstm_right)
    lstm_left_tmp = Dense(500, W_regularizer=l2)(lstm_left)

    lstm_right_tmp = keras.layers.BatchNormalization()(lstm_right_tmp)
    lstm_left_tmp = keras.layers.BatchNormalization()(lstm_left_tmp)

    lstm_right_tmp = Activation('relu')(lstm_right_tmp)
    lstm_left_tmp = Activation('relu')(lstm_left_tmp)

    lstm_right_tmp = keras.layers.Reshape((1, 500))(lstm_right_tmp)
    lstm_left_tmp = keras.layers.Reshape((1, 500))(lstm_left_tmp)

    ##### try lstm left/right instead of mem
    inputs_left3 = merge([lstm_right_tmp, inputs_left22, lstm_right_tmp], mode='concat', concat_axis=1)
    inputs_right3 = merge([lstm_left_tmp, inputs_right22, lstm_left_tmp], mode='concat', concat_axis=1)

    lstm_left2 = LSTM(1500, return_sequences=False, name='leftLSTM2', W_regularizer=l2)(inputs_left3)
    lstm_right2 = LSTM(1500, return_sequences=False, go_backwards=True, name='rightLSTM2', W_regularizer=l2)(inputs_right3)

    lstm_out = merge([lstm_left2, lstm_right2, lstm_right, lstm_left], mode='concat', concat_axis=1)




    lstm_out = Dense(output_size, W_regularizer=l2)(lstm_out)
    lstm_out = keras.layers.BatchNormalization()(lstm_out)
    lstm_out = Activation('tanh')(lstm_out)


    model = Model(input=[inputs_left, inputs_right, inputs_left_p, inputs_right_p], output=[lstm_out])
    return model
