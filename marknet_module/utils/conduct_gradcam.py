#特定のCNNに入力画像を受け取った状況でGrad_camによりその予測根拠を可視化する

import tensorflow as tf
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
from keras.layers.core import Lambda
from tensorflow.python.framework import ops
import keras
import cv2
from keras import backend as K
from keras.preprocessing import image
import numpy as np


from keras.applications.vgg16 import  VGG16, preprocess_input, decode_predictions



class Gradcam:

    def __init__(self,mode):
        self.mode = mode

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

    def CAM(self):
        # simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception

        import io
        import requests
        from PIL import Image
        from torchvision import models, transforms
        from torch.autograd import Variable
        from torch.nn import functional as F
        import numpy as np
        import cv2
        import pdb

        # input image
        LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
        IMG_URL = 'http://media.mlive.com/news_impact/photo/9933031-large.jpg'

        # networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
        model_id = 1
        if model_id == 1:
            net = models.squeezenet1_1(pretrained=True)
            finalconv_name = 'features'  # this is the last conv layer of the network
        elif model_id == 2:
            net = models.resnet18(pretrained=True)
            finalconv_name = 'layer4'
        elif model_id == 3:
            net = models.densenet161(pretrained=True)
            finalconv_name = 'features'

        net.eval()

        # hook the feature extractor
        features_blobs = []

        def hook_feature(module, input, output):
            features_blobs.append(output.data.cpu().numpy())

        net._modules.get(finalconv_name).register_forward_hook(hook_feature)

        # get the softmax weight
        params = list(net.parameters())
        weight_softmax = np.squeeze(params[-2].data.numpy())

        def returnCAM(feature_conv, weight_softmax, class_idx):
            # generate the class activation maps upsample to 256x256
            size_upsample = (256, 256)
            bz, nc, h, w = feature_conv.shape
            output_cam = []
            for idx in class_idx:
                cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
                cam = cam.reshape(h, w)
                cam = cam - np.min(cam)
                cam_img = cam / np.max(cam)
                cam_img = np.uint8(255 * cam_img)
                output_cam.append(cv2.resize(cam_img, size_upsample))
            return output_cam

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])

        response = requests.get(IMG_URL)
        img_pil = Image.open(io.BytesIO(response.content))
        img_pil.save('test.jpg')

        img_tensor = preprocess(img_pil)
        img_variable = Variable(img_tensor.unsqueeze(0))
        logit = net(img_variable)

        # download the imagenet category list
        classes = {int(key): value for (key, value)
                   in requests.get(LABELS_URL).json().items()}

        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.numpy()
        idx = idx.numpy()

        # output the prediction
        for i in range(0, 5):
            print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

        # generate class activation mapping for the top1 prediction
        CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

        # render the CAM and output
        print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0]])
        img = cv2.imread('test.jpg')
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        cv2.imwrite('CAM.jpg', result)


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


    def modify_backprop(self, model, name):
        """
        逆伝播の算出及び最適化
        :param model: Keras学習済みモデル
        :param name:
        :return:
        """
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

        nb_classes = 5  # 18
        target_layer = lambda x: self.target_category_loss(x, category_index, nb_classes)

        data_ = []
        for i in input_model.layers:
            if i.name == 'add_marge':
                print('add_marge',i)
                data_.append(i)

        # レイヤー指定
        # モデルの推論を実行すると、予測クラス以外の値は0になる
        x = input_model.layers[9].output
        print('input',input_model.layers[9].name)
        x = Lambda(target_layer, output_shape=self.target_category_loss_output_shape)(x)
        model = keras.models.Model(input_model.layers[0].input, x)

        data = []
        for i in model.layers:
            if i.name == 'prediction_branch':
                print('prediction',i)
                data.append(i)

        conv_output = model.layers[3].output #3
        print('layer4', conv_output)
        # print(conv_output =  [l for l in input_model.layers if l.name == layer_name][0].output)


        # 予測クラス以外の値は0になっている ・予測の損失
        # sumをとり予測クラスの値のみを抽出
        loss = KTF.sum(model.layers[9].output)
        print('nameeeeee final', model.layers[9].name)

        # 予測クラスの値から最後のconv層までの勾配を算出する関数を定義
        #
        grads = self.normalize(KTF.gradients(loss, conv_output)[0])
        gradient_function = KTF.function([model.layers[0].input], [conv_output, grads])

        # 定義した勾配計算用の関数で算出
        output, grads_val = gradient_function([image])
        output, grads_val = output[0, :], grads_val[0, :, :, :]

        # 最後のconv層のチャンネル毎に勾配の平均を算出し
        # かくチャンネルの重要度とする
        # GAP
        weights = np.mean(grads_val, axis=(0, 1))
        cam = np.ones(output.shape[0: 2], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * (output[:, :, i]) # 255*(output[:,:,i])

        cam = cv2.resize(cam, (300, 300))

        # 負の値を0に変換。処理はReluと同意
        cam = np.maximum(cam, 0)
        # 値を0-1に正規化
        heatmap = cam / np.max(cam)

        # Return to BGR [0..255] from the preprocessed image
        image = image[0, :]
        image -= np.min(image)
        image = np.minimum(image, 255)

        cam1 = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

        # ヒートマップと入力画像の重ね合わせ
        cam = np.float32(cam1) + np.float32(image)
        cam = 255 * cam / np.max(cam)

        # ヒートマップのみ
        cam1 = cv2.resize(cam1, (300, 300))
        cv2.imwrite('/Users/matsunagamasaaki/Desktop/gimage.png', image)

        cv2.imwrite('/Users/matsunagamasaaki/Desktop/gcam.png', cam1)
        return np.uint8(cam), heatmap, cam1


    def conduct_gradcam(self, model, img, image_shape, category=None):
        """
        GradCAMの実行関数
        :param model: Keras学習済みモデル
        :param img: 入力画像
        :param image_shape: 入力画像サイズ
        :return: None
        """

        img_original= cv2.resize(img, (image_shape[0],image_shape[1]))
        image_= cv2.resize(img_original, (300, 300))
        img = np.expand_dims(img_original, axis=0)

        # Keras VGG16の前処理用
        #img = preprocess_input(img)

        # モデルの推論
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions[:4])
        print('predictions',predictions)
        print('prediction category:', predicted_class)

        # GradCAMの実行
        cam, heatmap,cam1 = self.grad_cam(input_model=model,
                                          image=img,
                                          category_index=predicted_class,
                                          layer_name='pool1m')

        cam_ = cv2.resize(cam,(300,300))
        self.register_gradient()
        guided_model = self.modify_backprop(model, 'GuidedBackProp')
        saliency_fn = self.compile_saliency_function(guided_model)
        saliency = saliency_fn([img, 0])

        gradcam = saliency[0] * heatmap[..., np.newaxis]
        gradcam = cv2.resize(gradcam[0],(300,300))
        guided_gradcam = cv2.resize(self.deprocess_image(gradcam),(300,300))

        # 画像の接合
        im_h = cv2.hconcat([image_, cam_, cam1, guided_gradcam])

        cv2.imwrite('/Users/matsunagamasaaki/Documents/ipsj_v4/UTF8/image/gradcam.png', cam1)

        #cv2.imshow('Attention/ Class Activation Map'+'_'+str(category), im_h)

        return im_h