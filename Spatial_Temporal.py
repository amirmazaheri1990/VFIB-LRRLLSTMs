from __future__ import print_function
from keras.layers import Input, Embedding, LSTM, Dense, merge, Activation, Lambda
from keras.models import Model, model_from_json
import keras
from keras.layers.core import Merge
import numpy as np
import pickle
import os, glob
import theano.sandbox.cuda
theano.sandbox.cuda.use("gpu0")
import time, random
import sys
import scipy.io as pio
from keras import backend as k
import modules,dataloader
from keras.layers import BatchNormalization
from keras.layers.wrappers import TimeDistributed


def get_model_layers_weights(model):
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    return layer_dict

def main(options, rep_one_dimensions, mylambda):
    retrain = int(options)-1
    vis_rep_one_len = int(rep_one_dimensions)
    lambda_Rate = float(mylambda)
    main_path = "/media/amir/Data/DATASETS/MPII/FIB/codes/"

    sys.path.insert(0, main_path)


    experiment = 'Spatial_Temporal_ICCV_sum_single_l2_0001_'+rep_one_dimensions+'_'+mylambda

    stories_path = main_path + 'stories/'
    models_path = main_path + 'models/'

    data_path = main_path+'data_splited/'
    experiment_name = models_path+experiment
    logs_path = main_path+'logs/'+experiment+".txt"
    data_path = main_path + 'data_splited/'
    final_output_len = 890

    l2 = keras.layers.regularizers.l2(l = 0.0001)


    print('Building Model....')
    inputs_left = Input(shape=(10,),name='wordsLeft')
    inputs_right = Input(shape=(10,),name='wordsRight')
    inputs_visual = Input(shape=(14,14,512),name='visual')
    inputs_c3d_seq = Input(shape=(10,4096), name='visual_c3d')

    vis_rep_len = 4096


    vis_rep_one = keras.layers.Reshape((196,512))(inputs_visual)
    batchNorm_vinormOne = BatchNormalization(mode = 1, name = 'vis_rep_one_normalization')
    vis_rep_one = TimeDistributed(batchNorm_vinormOne)(vis_rep_one)
    visualInptShareDense = Dense(vis_rep_one_len, name = 'VisualInputShareDense', W_regularizer=l2)
    shareDenseVisual = TimeDistributed(visualInptShareDense)(vis_rep_one)
    shareDenseVisualNormalization = BatchNormalization(mode = 1, name = 'shareDenseVisualNormalization')
    shareDenseVisual = TimeDistributed(shareDenseVisualNormalization)(shareDenseVisual)
    shareDenseVisual = Activation('tanh')(shareDenseVisual)

    ###Question part
    text_encoder = modules.get_textEncoder_newIdea(vis_rep_one_len,l2)

    lstm_out = text_encoder([inputs_left, inputs_right])

    ###
    lstm_out_tmp = lstm_out
    lstm_out_Attn_spatial = Dense(200, name = 'lstm_out_Dense_spatial', W_regularizer=l2)(lstm_out_tmp)
    #lstm_out_Attn_temporal = lstm_out_Attn_spatial
    lstm_out_Attn_temporal = Dense(200, name = 'lstm_out_Dense_temporal', W_regularizer=l2)(lstm_out_tmp)




    #######C3D
    batchNorm_vinormOne = BatchNormalization(mode = 1, name = 'vis_rep_one_normalization')
    vis_rep_one = TimeDistributed(batchNorm_vinormOne)(inputs_c3d_seq)
    visualInptShareDense = Dense(vis_rep_one_len, name = 'VisualInputShareDense', W_regularizer=l2)
    ShareDenseC3D = TimeDistributed(visualInptShareDense)(vis_rep_one)
    ShareDenseC3DNormalization = BatchNormalization(mode = 1, name = 'ShareDenseC3DNormalization')
    ShareDenseC3D = TimeDistributed(ShareDenseC3DNormalization)(ShareDenseC3D)
    ShareDenseC3D = Activation('tanh')(ShareDenseC3D)


    DenseToProbabilities_C3D = Dense(1,name = 'DenseToProbabilitiesC3D', W_regularizer=l2)
    ShareDense_C3D = Dense(200, name = 'ShareDenseC3D2', W_regularizer=l2)
    ShareBatchNorm_h = BatchNormalization(mode=1, name='ShareBatchNorm_h')
    LSTMToC3DAttn = LSTM(16, name = 'SelectorLSTM',return_sequences = True, unroll = True, W_regularizer=l2)
    p_norm = BatchNormalization(mode = 1, name='pNorm_C3D')

    lstm_out3 = keras.layers.RepeatVector(10)(lstm_out_Attn_temporal) #196*10*D(512)
    ShareDenseC3D2 = TimeDistributed(ShareDense_C3D)(ShareDenseC3D)
    h = merge([lstm_out3, ShareDenseC3D2], mode = 'sum')
    h = TimeDistributed(ShareBatchNorm_h)(h)
    h = Activation('tanh')(h)
    p = keras.layers.wrappers.Bidirectional(LSTMToC3DAttn)(h)
    p = keras.layers.wrappers.TimeDistributed(DenseToProbabilities_C3D)(p)
    p = keras.layers.Reshape((10,))(p)
    p = p_norm(p)
    p_final_out = Activation('softmax')(p)
    Attn_C3D = merge([p_final_out, ShareDenseC3D],mode = 'dot', dot_axes = (1,1))#D

    #lstm_out_C3d = keras.layers.Highway(name = 'Highway_c3d', activation='tanh')(lstm_out)
    #Attn_C3D = merge([C3DAttn_out,lstm_out], mode = 'sum',name = 'C3DAttn')

    ####Attn

    shareDenseVisualAttn = Dense(200, name = 'shareDenseVisual_Attn', W_regularizer=l2)
    DenseToProbabilities_SAN = Dense(1,name = 'DenseToProbabilities_SAN', W_regularizer=l2)
    ShareBatchNormSAN_h = BatchNormalization(mode=1, name='ShareBatchNormSAN_h')
    pSAN_norm =  BatchNormalization(mode = 1, name='pNorm_SAN')

    lstm_out3 = keras.layers.RepeatVector(196)(lstm_out_Attn_spatial) #196*10*D(512)
    shareDenseVisual2 = TimeDistributed(shareDenseVisualAttn)(shareDenseVisual)
    h = merge([shareDenseVisual2, lstm_out3], mode = 'sum')
    h = TimeDistributed(ShareBatchNormSAN_h)(h)
    h = Activation('tanh')(h)
    p = keras.layers.wrappers.TimeDistributed(DenseToProbabilities_SAN)(h)
    p = keras.layers.Reshape((196,))(p)
    p = pSAN_norm(p)
    p_final_out = Activation('softmax')(p)
    Attn_SAN = merge([p_final_out,shareDenseVisual],mode = 'dot', dot_axes = (1,1))#D

    #lstm_out_SAN= keras.layers.Highway(name = 'Highway_SAN', activation='tanh')(lstm_out)
    #Attn_SAN = merge([Attn_SAN], mode = 'sum',name = 'SANAttn')

    #lstm_out_final = keras.layers.Highway(name = 'HighwayLast', activation='tanh')(lstm_out)


    combined = merge([Attn_C3D,Attn_SAN,lstm_out],mode= 'sum',concat_axis=1, name = 'combined')

    combined = keras.layers.Dropout(0.7)(combined)

    Answer = Dense(final_output_len, name = 'final', W_regularizer=l2)(combined)
    Answer = keras.layers.BatchNormalization(mode = 1)(Answer)
    Answer = Activation('softmax')(Answer)

    model = Model(input=[inputs_left,inputs_right,inputs_visual, inputs_c3d_seq], output=[Answer])
    adag = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-8, decay=0.00005)
    model.compile(optimizer= adag, loss=['categorical_crossentropy'],  metrics=['accuracy'])
    print('model compiled')




    print('data loaded!!!!!')


    json_string = model.to_json()
    open(experiment_name+'.json', 'w').write(json_string)



    print('model saved')


    saveweightingname = experiment_name+'.h5'
    try:
            if(retrain==0):
                    raise Exception('pffff!!!')
            model.load_weights(saveweightingname)
            #model = keras.models.load_model(saveweightingname)
            story = []
            print('Saved model is loaded, continue training')
            model.compile(optimizer= adag, loss=['categorical_crossentropy'],  metrics=['accuracy'])
    except:
            try:
                if(retrain==0):
                    os.remove(logs_path)
            except:
                pass
            story = []
            pass


    try:
        os.remove(logs_path)
    except:
        pass
    v = 9999.9999
    p = v
    patient = 0
    v = 0
    max_patient = 5
    patientToMoveToSGD = 1
    lr = 0.001
    samples_per_epoch = 296960
    for iteration in range(1, 1000):
            print(experiment)
            h = model.fit_generator(dataloader.gen('/media/amir/2TF/mpii/'),nb_worker=1,pickle_safe=False, nb_epoch=1, samples_per_epoch=samples_per_epoch, verbose=1,max_q_size=100 ,validation_data=dataloader.gen_val('/media/amir/2TF/mpii/'), nb_val_samples = 21689)
            losses = [h.history['loss'][0],h.history['val_loss'][0]]
            vloss = h.history['val_acc'][0]
            with open(logs_path, "a") as myfile:
                myfile.write("iteration: "+str(iteration)+" "+str(h.history['loss'][0])+" "+str(vloss)+" \n")
            if(v<=vloss):
                patient = 0
                v = vloss
                model.save_weights(saveweightingname,overwrite=True)
                #model.save(saveweightingname)
                print('better model saved!!!!!!')
            else:
                patient = patient + 1
            if(patient==patientToMoveToSGD):
                print('model recompiled!!!!')
                sgd = keras.optimizers.SGD(lr=0.001,nesterov=True)
                lr = lr/10
                adag = keras.optimizers.Adagrad(lr=lr, epsilon=1e-08)
                sgd = keras.optimizers.sgd(lr=lr, momentum=0.9, decay=0., nesterov=True)
                #model = keras.models.load_model(saveweightingname)
                model.load_weights(saveweightingname)
                samples_per_epoch = int(samples_per_epoch/2)
                model.compile(optimizer= adag, loss=['categorical_crossentropy'],  metrics=['accuracy'])
            if(patient>=max_patient):
                print('ended!!!!')
                break
            story.append(losses)

    #model.save_weights(saveweightingname,overwrite=True)


if __name__ == "__main__":
    random.seed()
    main('1', '7000', '2')
    # main(sys.argv[1], sys.argv[2], sys.argv[3])
    #main('2')
