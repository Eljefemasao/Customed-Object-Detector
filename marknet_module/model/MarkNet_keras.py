
from keras.models import Sequential
#from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Add, Input
from keras.layers import Dropout
from keras.models import Model
import keras 

# attentionで使用
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Multiply
from keras.layers import Add
from keras.layers import concatenate

import numpy as np

class Model_:

    def __init__(self, input_shape):
        self.input_shape = input_shape
    

    def MarkNet(self):

        # モデル
        net2 = {}
        # Attention Branch
        att = {}
        # Attention マップ
        att_map = {}

        inputs = Input(shape=self.input_shape)
        
        net2['input_m'] = inputs
        net2['conv1_1m'] = Convolution2D(32, 3,3,
                                        activation='relu',
                                        border_mode='same',
                                        name='conv1_1m')(net2['input_m'])

        net2['conv1_2m'] = Convolution2D(32, 3,3,
                                        activation='relu',
                                        border_mode='same',
                                        name='conv1_2m')(net2['conv1_1m'])

        net2['pool1m'] = MaxPooling2D((2,2),
                                   border_mode='same',
                                   name='pool1m')(net2['conv1_2m'])


        net2['drop1m'] = Dropout(0.25,
                                name='drop1m')(net2['pool1m'])


        net2['conv2_1m'] = Convolution2D(5, 3,3, # kernel was 128 for including to ssd
                                        activation='relu',
                                        border_mode='same',
                                        name='conv2_1m')(net2['drop1m'])
        

        net2['pool2m'] = MaxPooling2D((2,2),
                                   border_mode='same',
                                   name='pool2m')(net2['conv2_1m'])



        ### Attention Branch

        att['batch_norm'] = BatchNormalization(name='batch_norm')(net2['pool2m'])
        att['conv1_att'] = Convolution2D(16, 1,1,
                                       activation='relu',
                                       border_mode='same',
                                       name='conv1_att')(att['batch_norm'])
        att['conv2_att'] = Convolution2D(5, 1, 1,
                                         activation='relu',
                                         border_mode='same',
                                         name='conv2_att')(att['conv1_att'])

        att['gap'] = GlobalAveragePooling2D(name='gap')(att['conv2_att'])
        att['softmax'] = Activation(activation='softmax', name='attention_branch')(att['gap'])

        print('attention_branch:', np.shape(att['softmax']))
        ### Attention Map

        att_map['map'] = Convolution2D(5,1,1,
                                       border_mode='same',
                                       name='map')(att['conv1_att'])

        att_map['batch_norm'] = BatchNormalization(name='batch_norm_map')(att_map['map'])
        att_map['sigmoid'] = Activation(activation='sigmoid')(att_map['batch_norm'])

        ### Connection

        print("sssssssssssssssss",np.shape(net2['pool2m']))
        print(np.shape(att_map['sigmoid']))
        net2['marge'] = Multiply()([net2['pool2m'], att_map['sigmoid']])
        net2['add_marge'] = Add()([net2['marge'], net2['pool2m']])

        ###
      #  model.add(Conv2D(512, (3, 3), name='conv6_2')) #元々64ch
      #  model.add(Activation('relu'))
      #  model.add(MaxPooling2D(pool_size=(2, 2)))
        ###

        net2['drop2m'] = Dropout(0.25,
                                name='drop2m')(net2['add_marge'])
        

        net2['flat1m'] = Flatten(name='flat1m')(net2['drop2m'])
        net2['dense1m'] = Dense(512, name='dense1m', activation='relu')(net2['flat1m'])

        net2['drop3m'] = Dropout(0.5,
                                name='drop3m')(net2['dense1m'])
        
        net2['dense2m'] = Dense(5, name='prediction_branch', activation='softmax')(net2['drop3m'])

        net2['predictions'] = concatenate([net2['dense2m'],
                                          att['softmax']],
                                         axis=1, name='predictions')

        model = Model(inputs=net2['input_m'], outputs=net2['predictions'])

        
        return model





