#特定のCNNに入力画像を受け取った状況でGrad_camによりその予測根拠を可視化する

import tensorflow as tf
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
from keras.preprocessing import image
import keras

from keras.models import load_model
import random as rn
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_json
from keras.applications.vgg16 import VGG16, preprocess_input 
#https://stackoverflow.com/questions/47555829/preprocess-input-method-in-keras                                                                                                            

from PIL import Image
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd

from sklearn.decomposition import PCA
#from sklearn.externals import joblib
import joblib
from sklearn.svm import LinearSVC

from PIL import ImageFile
import math


from keras.preprocessing import image
from keras.preprocessing.image import array_to_img, img_to_array, load_img

from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
import keras.backend as K
import numpy as np
import keras 
import sys 
import cv2


import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.applications.resnet50 import (ResNet50, preprocess_input)
from keras.preprocessing import image


import sys
#from matplotlib_util import save_model_to_file


import ast
import numpy as np
import math
import random
from tensorflow.keras.preprocessing.image import img_to_array as img_to_array
from tensorflow.keras.preprocessing.image import load_img as load_img

STANDARD_SIZE=(300,300)


class Gradcam:

    def __init__(self,mode):
        self.mode = mode

    def img_to_matrix(self, filename, verbose=False):
        """
        parse image
        """
        img = Image.open(filename)

    #    if verbase:
    #       print('changing size from %s to %s'% (str(img.size), str(STANDARD_SIZE)))
        img = img.resize(STANDARD_SIZE)
        imgArray = np.asarray(img)
        return imgArray 


    def grad_cam(self, input_model, image, category_index, layer_name, boxcoords):
        
        nb_classes = 2#6#18

        # bounding box boords
        xmin=boxcoords[0]
        ymin=boxcoords[1]
        xmax=boxcoords[2]
        ymax=boxcoords[3]


        target_layer = lambda x: self.target_category_loss(x, category_index, nb_classes)

        # レイヤー指定
        x = input_model.layers[-3].output
        x = Lambda(target_layer, output_shape=self.target_category_loss_output_shape)(x)
        model = keras.models.Model(input_model.layers[0].input, x)

        conv_output =model.layers[-38].output #model.layers[5].output 
        #print(conv_output =  [l for l in input_model.layers if l.name == layer_name][0].output)

        loss = KTF.sum(model.layers[-3].output)

        grads = self.normalize(KTF.gradients(loss, conv_output)[0])
        gradient_function = KTF.function([model.layers[0].input], [conv_output, grads])
        output, grads_val = gradient_function([image])
        output, grads_val = output[0, :], grads_val[0, :, :, :]

        #多分GAP
        weights = np.mean(grads_val, axis = (0, 1))
        cam = np.ones(output.shape[0 : 2], dtype = np.float32)

        for i, w in enumerate(weights):
            cam += w*(255*output[:, :, i])

        cam = cv2.resize(cam, (300, 300))
        cam = np.maximum(cam, 0)
        heatmap = cam / np.max(cam)
        
        #Return to BGR [0..255] from the preprocessed image
        image = image[0, :]
        image -= np.min(image)
        image = np.minimum(image, 255)

        cam1 = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

        # boundingbox以外のヒートマップ領域を消す
        cam1 = cv2.resize(cam1, (720, 405))
        cam1[:,0:xmin, :] = 0 # left
        cam1[0:ymin,:,:] = 0 # top
        cam1[ymax:-1,:,:] = 0 # bottom
        cam1[:,xmax:-1,:] = 0
        print('PPPPPPPPP', np.shape(cam1))
        cam1 = cv2.resize(cam1, (300,300))


        heatmap = cv2.resize(heatmap, (720, 405))
        heatmap[:,0:xmin] = 0 # left
        heatmap[0:ymin,:] = 0 # top
        heatmap[ymax:-1,:] = 0 # bottom
        heatmap[:,xmax:-1] = 0
        heatmap = cv2.resize(heatmap, (300,300))



        # 視認性を上げるため入力画像を重ねる
        cam = np.float32(cam1) + np.float32(255*image)
        cam = 255 * cam / np.max(cam)

        return np.uint8(cam), heatmap

        
    def target_category_loss(self, x, category_index, nb_classes):
        return tf.multiply(x, KTF.one_hot([category_index], nb_classes))


    def target_category_loss_output_shape(self, input_shape):
        return input_shape


    def normalize(self, x):
        # utility function to normalize a tensor by its L2 norm
        return x / (KTF.sqrt(KTF.mean(KTF.square(x)) + 1e-5))


    def load_image(self, path):
        img_path = path
        img = image.load_img(img_path, target_size=(300, 300))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
    #   x = preprocess_input(x)
        return x


    def register_gradient(self):
        if "GuidedBackProp" not in ops._gradient_registry._registry:
            @ops.RegisterGradient("GuidedBackProp")
            def _GuidedBackProp(op, grad):
                dtype = op.inputs[0].dtype
                return grad * tf.cast(grad > 0., dtype) *  tf.cast(op.inputs[0] > 0., dtype)


    def compile_saliency_function(self, model, activation_layer='conv7_2'):

        input_img = model.input
        layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
        layer_output = layer_dict[activation_layer].output
        #各特徴マップの中より予測により影響を与えているものを選別
        max_output = K.max(layer_output, axis=3)
        saliency = K.gradients(K.sum(max_output), input_img)[0]
        return K.function([input_img, K.learning_phase()], [saliency])
        
        #return KTF.function([input_img, KTF.learning_phase()], [saliency])

    #guided backpropっぽい
    def modify_backprop(self, model, name):

        g = tf.get_default_graph()

        with g.gradient_override_map({'Relu': name}):

            # get layers that have an activation                              # 'hasttr() = activation'属性を持っているのかTrue of Falseで返す
            layer_dict = [layer for layer in model.layers[1:] if hasattr(layer, 'activation')]
            print(layer_dict)
            # replace relu activation
            for layer1 in layer_dict:
                if layer1.activation == keras.activations.relu:
                    layer1.activation = tf.nn.relu

            # re-instanciate a new model
            new_model = model
        return new_model


    def deprocess_image(self, x):
        '''
        Same normalization as in:
        https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
        '''
        if np.ndim(x) > 3:
            x = np.squeeze(x)
        # normalize tensor: center on 0., ensure std is 0.1
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255
        if KTF.image_dim_ordering() == 'th':
            x = x.transpose((1, 2, 0))
        # 0~255に正規化    
        x = np.clip(x, 0, 255).astype('uint8')
        return x


    def conduct_gradcam(self, model, img, score, boxcoords, label_list, box_color, label_name):

    #    Grad-CAMの実行
#        img = image.img_to_array(img_to_matrix(url))
#        img.astype('float32')
       # img /= 255.0
        img = cv2.resize(img, (300,300))
        img = np.expand_dims(img,axis=0)
        # preprocessed_input = load_image("/home/seimei/image_0058_brush.jpg")
        predictions=model.predict(img)#np.expand_dims(img, axis=0))
        predicted_class= np.argmax(label_list)# np.argmax(predictions)
        print('score', score)
        print(label_list)
        print(predicted_class)
        # モデルのレイヤ確認
        for i in model.layers:
            print('ssssssss',i.name)

        cam, heatmap = self.grad_cam(model, img, predicted_class, "conv7_2", boxcoords)
        cam_ = cv2.resize(cam,(720,405))

        # bounding box boords
        xmin=boxcoords[0]
        ymin=boxcoords[1]
        xmax=boxcoords[2]
        ymax=boxcoords[3]

        # boundingbox 表示        
        cam_ = cv2.rectangle(cam_, (xmin, ymin), (xmax, ymax), (85, 255, 0), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        #center = (int(round(xmax - xmin + 1) / 2 + xmin), int(round(ymax - ymin + 1) / 2 + ymin))
        top = (int(round(xmin)),int(ymin-1))
        cv2.putText(cam_, label_name, top, font, 0.8, (0,255,0), 2, cv2.LINE_AA)

        cv2.imshow("cam",cam_)
        heatmap_ = cv2.resize(heatmap,(720,405))


        resized_img =  cv2.resize(img[0], (720,405))
        self.register_gradient()
        guided_model = self.modify_backprop(model, 'GuidedBackProp')
        saliency_fn = self.compile_saliency_function(guided_model)

        saliency = saliency_fn([img,0])#[np.expand_dims(img, axis=0), 0])

#        cv2.imshow("Backpropagation", saliency)
        gradcam = saliency[0] * heatmap[..., np.newaxis] 
        gradcam = cv2.resize(gradcam[0],(720,405))

#        cv2.imshow('+backprop', gradcam)
        guided_gradcam = cv2.resize(self.deprocess_image(gradcam),(720, 405))#+ np.uint8(reised_img),(720,405))
        cv2.imshow("guided_gradcam", guided_gradcam)



            