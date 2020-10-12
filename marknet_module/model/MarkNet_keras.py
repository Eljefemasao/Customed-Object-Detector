
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Add, Input
import keras 



class Model:

    def __init__(self, input_shape):
        self.input_shape = input_shape

    def MarkNet(self):
         # モデル
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=self.input_shape ))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding='same')) #元々64ch
        model.add(Activation('relu'))
        
        model.add(Conv2D(256, (3, 3), name='conv8_2')) #元々64ch
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        ###
      #  model.add(Conv2D(512, (3, 3), name='conv6_2')) #元々64ch
      #  model.add(Activation('relu'))
      #  model.add(MaxPooling2D(pool_size=(2, 2)))
        ###
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2))
        model.add(Activation('softmax'))

        return model





