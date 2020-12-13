import keras.backend as K
from keras.engine.topology import Layer
import numpy as np
import cv2
from numpy import copy
import tensorflow as tf


def convert_to_binary_img(thresh):
    """
    reluの実現を行う関数
    255:白を0へ変換
    0:黒を１へ変換
    """
    # バイナリの光反射画像
    newArray = copy(thresh)
    d = {1: 0, 0:1}
    for k, v in d.items():
        newArray[thresh == k] = v

    newArray = np.expand_dims(newArray, axis=2)
    newArray =np.concatenate([newArray,newArray,newArray,newArray,newArray], axis=2)

    return newArray.astype('float32')


def detect_specular_reflectance(img, fmap):
    """
    paper: A video stream processor for real-time detection and correction of specular reflections in endoscopic images
    paper: http://homepages.laas.fr/parra/NEWCAS-TAISA08_PROCEEDINGS/PAPERS/13.pdf

    code: https://github.com/muratkrty/specularity-removal
    """
    from specularity_removal import specularity as spc

    feature_map_shape = fmap.shape
    img = cv2.resize(img, (feature_map_shape[1], feature_map_shape[2])) * 255

    # convert image BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('./tmp_img/input.png', img )

    # グレイスケールに変換
    #gray_img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r_img = m_img = np.array(gray_img)

    rimg = spc.derive_m(img, r_img)
    s_img = spc.derive_saturation(img, rimg)
    spec_mask = spc.check_pixel_specularity(rimg, s_img)
    enlarged_spec = spc.enlarge_specularity(spec_mask)
    cv2.imwrite('./tmp_img/enlarged.png', enlarged_spec)
    # 255:白を0
    # 0:黒を1に変換
    result = convert_to_binary_img(enlarged_spec)
    return result

def create_binary_map(img, fmap):
    """
    単純なマスク画像の生成
    convert gray-scale to binary image 
    """

    feature_map_shape = fmap.shape
    img = cv2.resize(img, (feature_map_shape[1], feature_map_shape[2])) * 255
    # cv2.imwrite('./tmp_img/input.png',img*255)

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

    # 255:白を0
    # 0:黒を1に変換
    result = convert_to_binary_img(thresh)

    return result


def binary_map_tensor_func(img4d, fmap):
    results = []
    for img3d in img4d:
        rimg3d = detect_specular_reflectance(img3d, fmap)
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





