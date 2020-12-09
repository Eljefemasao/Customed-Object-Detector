import keras.backend as K
from keras.engine.topology import Layer
import numpy as np
import cv2
from numpy import copy
import tensorflow as tf



def create_binary_map(img, fmap):

    feature_map_shape = fmap.shape
    img = cv2.resize(img, (feature_map_shape[1], feature_map_shape[2]))*255
    #cv2.imwrite('./tmp_img/input.png',img*255)

    # convert image BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # グレースケール
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # 閾値の設定
    thresh = cv2.threshold(blurred, 130, 255, cv2.THRESH_BINARY)[1]
    #cv2.imwrite('./tmp_img/thresh.png',thresh)

    # ノイズを消す
    # thresh = cv2.erode(thresh, None, iterations=2)
    # thresh = cv2.dilate(thresh, None, iterations=4)

    # バイナリの光反射画像
    newArray = copy(thresh)
    d = {1: 0, 0:1}
    for k, v in d.items():
        newArray[thresh == k] = v

    newArray = np.expand_dims(newArray, axis=2)
    newArray =np.concatenate([newArray,newArray,newArray,newArray,newArray], axis=2)

    return newArray.astype('float32')

def binary_map_tensor_func(img4d, fmap):
    results = []
    for img3d in img4d:
        rimg3d = create_binary_map(img3d, fmap)
        results.append(np.expand_dims(rimg3d, axis=0))
    return np.concatenate(results, axis=0)




class WhiteAttenuation(Layer):
    """
    attenuate white out region on image
    """

    def __init__(self, **kwargs):
        if K.image_dim_ordering() == 'tf':
            self.axis = 3
        else:
            self.axis = 1
        #self.input_img = input_img
        super(WhiteAttenuation, self).__init__(**kwargs)

    def call(self, x, mask=None):
        input_img, feature_map = x
        xout = tf.py_func( binary_map_tensor_func,
                           [input_img, feature_map],
                           'float32',
                           stateful=False,
                           name='binary_map')

        xout = K.stop_gradient(xout)

        xout.set_shape([feature_map.shape[0], feature_map.shape[1], feature_map.shape[2], 5])
        return xout


    def compute_output_shape( self, x) :
        return (None, 75, 75,  5)





