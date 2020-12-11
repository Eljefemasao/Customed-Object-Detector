#特定のCNNに入力画像を受け取った状況でGrad_camによりその予測根拠を可視化する

import tensorflow as tf
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
from PIL import Image
from keras.layers.core import Lambda
from tensorflow.python.framework import ops
import keras
import cv2
from keras import backend as K
from keras.preprocessing import image
import numpy as np


STANDARD_SIZE=(300,300)


class Gradcam:

    def __init__(self,mode):
        self.mode = mode

    def img_to_matrix(self, filename, verbose=False):
        """
        parse image
        """
        img = Image.open(filename)
        img = img.resize(STANDARD_SIZE)
        img_array = np.asarray(img)
        return img_array

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

    def compile_saliency_function(self, model, activation_layer='pool1m'):

        input_img = model.input
        layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
        layer_output = layer_dict[activation_layer].output
        #各特徴マップの中より予測により影響を与えているものを選別
        max_output = K.max(layer_output, axis=3)
        saliency = K.gradients(K.sum(max_output), input_img)[0]
        return K.function([input_img, K.learning_phase()], [saliency])
        #return KTF.function([input_img, KTF.learning_phase()], [saliency])

    #guided backprop
    def modify_backprop(self, model, name):

        g = tf.get_default_graph()

        with g.gradient_override_map({'Relu': name}):

            # get layers that have an activation
            # 'hasttr() = activation'属性を持っているのかTrue of Falseで返す
            layer_dict = [layer for layer in model.layers[1:] if hasattr(layer, 'activation')]
        
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

    def grad_cam(self, input_model, image, category_index, layer_name):

        nb_classes = 10  # 18
        target_layer = lambda x: self.target_category_loss(x, category_index, nb_classes)

        # レイヤー指定
        x = input_model.layers[-1].output
        x = Lambda(target_layer, output_shape=self.target_category_loss_output_shape)(x)
        model = keras.models.Model(input_model.layers[0].input, x)

        conv_output = model.layers[4].output  # model.layers[5].output
        print(conv_output)
        # print(conv_output =  [l for l in input_model.layers if l.name == layer_name][0].output)

        loss = KTF.sum(model.layers[8].output)
        print("~~~~~~")

        grads = self.normalize(KTF.gradients(loss, conv_output)[0])
        gradient_function = KTF.function([model.layers[0].input], [conv_output, grads])
        output, grads_val = gradient_function([image])
        output, grads_val = output[0, :], grads_val[0, :, :, :]

        # GAP
        weights = np.mean(grads_val, axis=(0, 1))
        cam = np.ones(output.shape[0: 2], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * (255 * output[:, :, i])

        cam = cv2.resize(cam, (300, 300))
        cam = np.maximum(cam, 0)
        heatmap = cam / np.max(cam)

        # Return to BGR [0..255] from the preprocessed image
        image = image[0, :]
        image -= np.min(image)
        image = np.minimum(image, 255)

        cam1 = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        cam = np.float32(cam1) + np.float32(image)
        cam = 255 * cam / np.max(cam)
        cam1 = cv2.resize(cam1, (300, 300))
        cv2.imshow('cam_', cam1)

        return np.uint8(cam), heatmap, cam1


    def conduct_gradcam(self, model, img, image_shape):

    #    Grad-CAMの実行
        img_original= cv2.resize(img, (image_shape[0],image_shape[1]))
        image_=cv2.resize(img_original, (300, 300))
        cv2.imshow('original', image_)

        img = np.expand_dims(img_original,axis=0)
        predictions=model.predict(img)

        predicted_class= np.argmax(predictions[:4])

        cam, heatmap,cam1 = self.grad_cam(model, img, predicted_class, 'pool1m')
        cam_ = cv2.resize(cam,(300,300))
        cv2.imshow("cam", cam_)

        heatmap_ = cv2.resize(heatmap,(300,300))
        cv2.imshow("heatmap",heatmap_)

        self.register_gradient()
        guided_model = self.modify_backprop(model, 'GuidedBackProp')
        saliency_fn = self.compile_saliency_function(guided_model)
        saliency = saliency_fn([img, 0])

        gradcam = saliency[0] * heatmap[..., np.newaxis]
        gradcam = cv2.resize(gradcam[0],(300,300))
        print(np.shape(gradcam))

        cv2.imshow('+backprop', gradcam)
        guided_gradcam = cv2.resize(self.deprocess_image(gradcam),(300,300))
        cv2.imshow("guided_gradcam", guided_gradcam)

        # 画像の接合
        im_h = cv2.hconcat([image_, cam_, cam1, guided_gradcam])
        cv2.imshow('Attention/ Class Activation Map', im_h)
#        cv2.waitKey(1)
