
from keras.models import Sequential
#from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Add, Input
from keras.layers import Dropout
from keras.models import Model
import keras 



class Model_:

    def __init__(self, input_shape):
        self.input_shape = input_shape
    

    def MarkNet(self):

        # モデル
        net2 = {}
        
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

        net2['conv2_1m'] = Convolution2D(64, 3,3, # kernel was 128 for including to ssd
                                        activation='relu',
                                        border_mode='same',
                                        name='conv2_1m')(net2['drop1m'])
        
        net2['conv2_2m'] = Convolution2D(64, 3,3, # kernel was 256 for including to ssd
                                        activation='relu',
                                        border_mode='same',
                                        name='conv8_2')(net2['conv2_1m']) # previous 64ch

        net2['pool2m'] = MaxPooling2D((2,2),
                                   border_mode='same',
                                   name='pool2m')(net2['conv2_2m'])
        ###
      #  model.add(Conv2D(512, (3, 3), name='conv6_2')) #元々64ch
      #  model.add(Activation('relu'))
      #  model.add(MaxPooling2D(pool_size=(2, 2)))
        ###

        net2['drop2m'] = Dropout(0.25,
                                name='drop2m')(net2['pool2m'])
        
        net2['flat1m'] = Flatten(name='flat1m')(net2['drop2m'])
        net2['dense1m'] = Dense(512, name='dense1m', activation='relu')(net2['flat1m'])

        net2['drop3m'] = Dropout(0.5,
                                name='drop3m')(net2['dense1m'])
        
        net2['dense2m'] = Dense(5, name='dense2m', activation='softmax')(net2['drop3m'])

        model = Model(net2['input_m'], net2['dense2m'])

        
        return model





