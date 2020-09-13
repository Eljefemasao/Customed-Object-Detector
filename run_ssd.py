
# KerasによるSSD(Single Shot Multi Box Detector)の実装
# https://github.com/rykov8/ssd_keras.git

import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import pickle
from random import shuffle
from scipy.misc import imread

from scipy.misc import imresize
import tensorflow as tf

from ssd import SSD300
from ssd_training import MultiboxLoss
from ssd_utils import BBoxUtility

from conduct_gradcam import Gradcam


plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'

np.set_printoptions(suppress=True)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))

# some constants
NUM_CLASSES = 6
input_shape = (300, 300, 3)


priors = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))
bbox_util = BBoxUtility(NUM_CLASSES, priors)

test1 = pickle.load(open('/Users/matsunagamasaaki/MasterResearch/ssd_keras/VOC_piece1.pkl','rb'))
#test2 = pickle.load(open('/Users/matsunagamasaaki/MasterResearch/ssd_keras/VOC_piece2.pkl','rb'))
#test3 = pickle.load(open('/Users/matsunagamasaaki/MasterResearch/ssd_keras/VOC_piece3.pkl','rb'))
#test4 = pickle.load(open('/Users/matsunagamasaaki/MasterResearch/ssd_keras/VOC_piece4.pkl','rb'))



#test1.update(test2)
#test1.update(test3)
#test1.update(test4)

print(len(test1))



gt = test1
keys = sorted(gt.keys())
num_train = int(round(0.8 * len(keys)))
train_keys = keys[:num_train]
val_keys = keys[num_train:]
num_val = len(val_keys)


class Generator(object):
    def __init__(self, gt, bbox_util,
                 batch_size, path_prefix,
                 train_keys, val_keys, image_size,
                 saturation_var=0.5,
                 brightness_var=0.5,
                 contrast_var=0.5,
                 lighting_std=0.5,
                 hflip_prob=0.5,
                 vflip_prob=0.5,
                 do_crop=True,
                 crop_area_range=[0.75, 1.0],
                 aspect_ratio_range=[3. / 4., 4. / 3.]):
        self.gt = gt
        self.bbox_util = bbox_util
        self.batch_size = batch_size
        self.path_prefix = path_prefix
        self.train_keys = train_keys
        self.val_keys = val_keys
        self.train_batches = len(train_keys)
        self.val_batches = len(val_keys)
        self.image_size = image_size
        self.color_jitter = []
        if saturation_var:
            self.saturation_var = saturation_var
            self.color_jitter.append(self.saturation)
        if brightness_var:
            self.brightness_var = brightness_var
            self.color_jitter.append(self.brightness)
        if contrast_var:
            self.contrast_var = contrast_var
            self.color_jitter.append(self.contrast)
        self.lighting_std = lighting_std
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.do_crop = do_crop
        self.crop_area_range = crop_area_range
        self.aspect_ratio_range = aspect_ratio_range

    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])

    def saturation(self, rgb):
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * self.saturation_var
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)

    def brightness(self, rgb):
        alpha = 2 * np.random.random() * self.brightness_var
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha
        return np.clip(rgb, 0, 255)

    def contrast(self, rgb):
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * self.contrast_var
        alpha += 1 - self.contrast_var
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 255)

    def lighting(self, img):
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigvec.dot(eigval * noise) * 255
        img += noise
        return np.clip(img, 0, 255)

    def horizontal_flip(self, img, y):
        if np.random.random() < self.hflip_prob:
            img = img[:, ::-1]
            y[:, [0, 2]] = 1 - y[:, [2, 0]]
        return img, y

    def vertical_flip(self, img, y):
        if np.random.random() < self.vflip_prob:
            img = img[::-1]
            y[:, [1, 3]] = 1 - y[:, [3, 1]]
        return img, y

    def random_sized_crop(self, img, targets):
        img_w = img.shape[1]
        img_h = img.shape[0]
        img_area = img_w * img_h
        random_scale = np.random.random()
        random_scale *= (self.crop_area_range[1] -
                         self.crop_area_range[0])
        random_scale += self.crop_area_range[0]
        target_area = random_scale * img_area
        random_ratio = np.random.random()
        random_ratio *= (self.aspect_ratio_range[1] -
                         self.aspect_ratio_range[0])
        random_ratio += self.aspect_ratio_range[0]
        w = np.round(np.sqrt(target_area * random_ratio))
        h = np.round(np.sqrt(target_area / random_ratio))
        if np.random.random() < 0.5:
            w, h = h, w
        w = min(w, img_w)
        w_rel = w / img_w
        w = int(w)
        h = min(h, img_h)
        h_rel = h / img_h
        h = int(h)
        x = np.random.random() * (img_w - w)
        x_rel = x / img_w
        x = int(x)
        y = np.random.random() * (img_h - h)
        y_rel = y / img_h
        y = int(y)
        img = img[y:y + h, x:x + w]
        new_targets = []
        for box in targets:
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            if (x_rel < cx < x_rel + w_rel and
                    y_rel < cy < y_rel + h_rel):
                xmin = (box[0] - x_rel) / w_rel
                ymin = (box[1] - y_rel) / h_rel
                xmax = (box[2] - x_rel) / w_rel
                ymax = (box[3] - y_rel) / h_rel
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(1, xmax)
                ymax = min(1, ymax)
                box[:4] = [xmin, ymin, xmax, ymax]
                new_targets.append(box)
        new_targets = np.asarray(new_targets).reshape(-1, targets.shape[1])
        return img, new_targets

    def generate(self, train=True):
        while True:
            if train:
                shuffle(self.train_keys)
                keys = self.train_keys
            else:
                shuffle(self.val_keys)
                keys = self.val_keys
            inputs = []
            targets = []
            for key in keys:
                img_path = self.path_prefix + key
                img = imread(img_path).astype('float32')
                y = self.gt[key].copy()
                if train and self.do_crop:
                    img, y = self.random_sized_crop(img, y)
                img = imresize(img, self.image_size).astype('float32')
                if train:
                    shuffle(self.color_jitter)
                    for jitter in self.color_jitter:
                        img = jitter(img)
                    if self.lighting_std:
                        img = self.lighting(img)
                    if self.hflip_prob > 0:
                        img, y = self.horizontal_flip(img, y)
                    if self.vflip_prob > 0:
                        img, y = self.vertical_flip(img, y)
                y = self.bbox_util.assign_boxes(y)
                inputs.append(img)
                targets.append(y)
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    yield preprocess_input(tmp_inp), tmp_targets


path_prefix = ''#'../../frames/'
gen = Generator(gt, bbox_util, 8, '',
                train_keys, val_keys,
                (input_shape[0], input_shape[1]), do_crop=False)


model = SSD300(input_shape, num_classes=NUM_CLASSES)
#model.load_weights('/Users/matsunagamasaaki/MasterResearch/ssd_keras/weights_SSD300.hdf5', by_name=True)

#これがメイン 0904
model.load_weights('/Users/matsunagamasaaki/MasterResearch/ssd_keras/model/my_model_weights2.hdf5', by_name=True)

#model.load_weights('/Users/matsunagamasaaki/MasterResearch/ssd_keras/model/20200425_6k.hdf5', by_name=True)

#model.load_weights('/Users/matsunagamasaaki/MasterResearch/ssd_keras/checkpoints/weights.10-1.62.hdf5', by_name=True)


#model.load_weights('/Users/matsunagamasaaki/MasterResearch/ssd_keras/checkpoints/weights.04-0.74.hdf5', by_name=True)



freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
          'conv2_1', 'conv2_2', 'pool2',
          'conv3_1', 'conv3_2', 'conv3_3', 'pool3']#,
#           'conv4_1', 'conv4_2', 'conv4_3', 'pool4'
for L in model.layers:
    if L.name in freeze:
        L.trainable = False


base_lr = 3e-4
optim = keras.optimizers.Adam(lr=base_lr)
# optim = keras.optimizers.RMSprop(lr=base_lr)
# optim = keras.optimizers.SGD(lr=base_lr, momentum=0.9, decay=decay, nesterov=True)
model.compile(optimizer=optim,
              loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss)


def schedule(epoch, decay=0.9):
    return base_lr * decay**(epoch)

callbacks = [keras.callbacks.ModelCheckpoint('./checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                             verbose=1,
                                             save_weights_only=True),
             keras.callbacks.LearningRateScheduler(schedule)]

import cv2
import glob

#美術品
path = sorted(glob.glob('/Users/matsunagamasaaki/MasterResearch/fragment_annotation/data5/data/test_frames/*.jpg'))


#print(keys)
import re
from collections import OrderedDict


def sortedStringList(array=[]):
    sortDict = OrderedDict()
    for splitList in array:
        sortDict.update({splitList: [int(x) for x in re.split("(\d+)", splitList) if bool(re.match("\d*", x).group())]})
    return [sortObjKey for sortObjKey, sortObjValue in sorted(sortDict.items(), key=lambda x: x[1])]


def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


path_ = sortedStringList(path)
test5 = pickle.load(open('/Users/matsunagamasaaki/MasterResearch/ssd_keras/VOC_piece1.pkl', 'rb'))
keys = test5.keys()
ground_truth = test5.values()




from timeit import default_timer as timer




accum_time = 0
curr_fps = 0
fps = "FPS: " + str(curr_fps)
prev_time = timer()

for idx, _ in zip(keys, ground_truth):

    gt_ = test5[idx]
    inputs = []
    images = []
    img_path = idx  #中身"/Users/matsunagamasaaki/MasterResearch/fragment_annotation/data5/data/frames_in/0/0/44.jpg"
    img = image.load_img(img_path, target_size=(300, 300))

    img = image.img_to_array(img)
#    img = cv2.resize(img, (300, 300))
    print(img)
    images.append(imread(img_path))
    inputs.append(img.copy())
    inputs = preprocess_input(np.array(inputs))

    # モデルによるbboxの推論
    preds = model.predict(inputs, batch_size=1, verbose=1)
    results = bbox_util.detection_out(preds)

    voc_classes = ["background", "A", "B", "C", "D", "E"]
    result = {'A': [], 'B': [], 'C': [], 'D': [], 'E': []}



    for i in gt_:
        if list(i[4:]) == [0,1,0,0,0,0]:
            label = voc_classes[0]
            result[label] = i[:4]
        elif list(i[4:]) == [0,0,1,0,0,0]:
            label = voc_classes[1]
            result[label] = i[:4]
        elif list(i[4:]) == [0,0,0,1,0,0]:
            label = voc_classes[2]
            result[label] = i[:4]
        elif list(i[4:]) == [0,0,0,0,1,0]:
            label = voc_classes[3]
            result[label] = i[:4]
        elif list(i[4:]) == [0,0,0,0,0,1]:
            label = voc_classes[4]
            result[label] = i[:4]
    
    """ 検出精度の算出
    dirname = '/Users/matsunagamasaaki/MasterResearch/Object-Detection-Metrics/groundtruths/' + \
              idx.split('/')[-1].split('.')[0] + '.txt'
    with open(dirname, 'a') as f:
        for i in result.keys():
            try:
                f.write(str(i) + ' ' + str(int(round(result[i][0]*img.shape[1]))) + ' ' + str(int(round(result[i][1]*img.shape[0]))) + ' ' +
                        str(int(round(result[i][2]*img.shape[1]))) + ' ' + str(int(round(result[i][3]*img.shape[0]))) + '\n')
            except IndexError:
                pass
    """

    det_label = results[0][:, 0]
    det_conf = results[0][:, 1]
    det_xmin = results[0][:, 2]
    det_ymin = results[0][:, 3]
    det_xmax = results[0][:, 4]
    det_ymax = results[0][:, 5]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.5]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    gt_label = gt_[:, 4:]
    print('ttttttttttttttttt', gt_label)
    gt_xmin = gt_[:, 0]
    gt_ymin = gt_[:, 1]
    gt_xmax = gt_[:, 2]
    gt_ymax = gt_[:, 3]

    detection = {'background': [], 'A': [], 'B': [], 'C': [], 'D': [], 'E': []}


    # hsv color
    colors = plt.cm.hsv(np.linspace(0, 1, 4)).tolist()
    currentAxis = plt.gca()

    for i in range(top_conf.shape[0]):

        xmin = int(round(top_xmin[i] * img.shape[1]))
        ymin = int(round(top_ymin[i] * img.shape[0]))
        xmax = int(round(top_xmax[i] * img.shape[1]))
        ymax = int(round(top_ymax[i] * img.shape[0]))
        score = top_conf[i]
        voc_classes = ["background","A", "B", "C", "D", "E"]
        label = int(top_label_indices[i])
        print(top_label_indices)
        label_name = voc_classes[label]


        display_txt = '{:0.2f}, {}'.format(score, label_name)
        coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
        # currentAxis.add_patch(plt.Rectangle(*coords,fill=False, edgecolor='white',linewidth=2))

        g_xmin = [int(round(i)) for i in gt_xmin * img.shape[1]]
        g_ymin = [int(round(i)) for i in gt_ymin * img.shape[0]]
        g_xmax = [int(round(i)) for i in gt_xmax * img.shape[1]]
        g_ymax = [int(round(i)) for i in gt_ymax * img.shape[0]]

        try:
            detection[label_name] = [score, g_xmin[label], g_ymin[label], g_xmax[label], g_ymax[label]]
 
        except IndexError:
            pass


    for i, img, in enumerate(images):
        # Parse the outputs.
        det_label = results[i][:, 0]
        det_conf = results[i][:, 1]
        det_xmin = results[i][:, 2]
        det_ymin = results[i][:, 3]
        det_xmax = results[i][:, 4]
        det_ymax = results[i][:, 5]



        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.5]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]



        gt_label = gt_[:,4:]
        print('ttttttttttttttttt', gt_label)
        gt_xmin = gt_[:,0]
        gt_ymin = gt_[:,1]
        gt_xmax = gt_[:,2]
        gt_ymax = gt_[:,3]



        colors = plt.cm.hsv(np.linspace(0, 1, 4)).tolist()

        currentAxis = plt.gca()

        for i in range(top_conf.shape[0]):

            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))
            score = top_conf[i]
            voc_classes = ["background","A", "B", "C", "D", "E"]

            label = int(top_label_indices[i])
            print(top_label_indices)
            label_name = voc_classes[label]

            if label_name is "pieceA":
                box_color = (0, 255, 0)
            elif "pieceB":
                box_color = (0, 255, 255)
            elif "pieceC":
                box_color = (255, 255, 0)
            elif "pieceD":
                box_color = (0, 0, 255)
            elif "pieceE":
                box_color = (255, 0, 0)

            display_txt = '{:0.2f}, {}'.format(score, label_name)
            coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
            # currentAxis.add_patch(plt.Rectangle(*coords,fill=False, edgecolor='white',linewidth=2))


            g_xmin=[int(round(i)) for i in gt_xmin * img.shape[1]]
            g_ymin=[int(round(i)) for i in gt_ymin * img.shape[0]]
            g_xmax=[int(round(i)) for i in gt_xmax * img.shape[1]]
            g_ymax=[int(round(i)) for i in gt_ymax * img.shape[0]]



            img = cv2.resize(img, (720, 405))
#            vo.track(img, img_id)
#            matched_kps_signal = [img_id, vo.num_matched_kps]
#            print('ssssssssssssssssssssssssssssssssss', matched_kps_signal)
#            print(xmin)

            # draw groudtruth box 
#            for i in range(len(g_xmin)):
#                img=cv2.rectangle(img, (g_xmin[i], g_ymin[i]), (g_xmax[i], g_ymax[i]), (255, 0, 0), 1)



            imgg = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (85, 255, 0), 1)

            font = cv2.FONT_HERSHEY_SIMPLEX

            ssd_center = (int(round(xmax - xmin + 1) / 2 + xmin), int(round(ymax - ymin + 1) / 2 + ymin))
            cv2.putText(img, label_name, ssd_center, font, 0.4, box_color, 1, cv2.LINE_AA)
            import math

            n = 3  # 切り捨てしたい桁
            y = math.floor(score * 10 ** n) / (10 ** n)

            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps +1

            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0


            cv2.putText(img, str(y), (xmin, ymin - 2), font, 0.5, box_color, 1, cv2.LINE_AA)
            cv2.putText(img, fps, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (85, 255, 0), 1)


            imgg = pil2cv(imgg)
            gradcam = Gradcam(mode = 'dd')
            gradcam.conduct_gradcam(model,img)

            cv2.imshow('test', imgg)
 #           cv2.imwrite('/Users/matsunagamasaaki/MasterResearch/annotation_hidefumi/tmp/'+str(idx)+str(i)+'.jpg', imgg)
            print('yes')
            cv2.waitKey(1)





