
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
import cv2
import glob
import re
from collections import OrderedDict
from timeit import default_timer as timer
import math


plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'

np.set_printoptions(suppress=True)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))

# some constants
NUM_CLASSES = 2#6
input_shape = (300, 300, 3)


priors = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))
bbox_util = BBoxUtility(NUM_CLASSES, priors)


# モデルと学習済み重みの呼び出し
model = SSD300(input_shape, num_classes=NUM_CLASSES)
#model.load_weights('/Users/matsunagamasaaki/MasterResearch/ssd_keras/weights_SSD300.hdf5', by_name=True)
#これがメイン 0904
#model.load_weights('/Users/matsunagamasaaki/MasterResearch/ssd_keras/model/my_model_weights2.hdf5', by_name=True)
model.load_weights('./model/cup.hdf5')


base_lr = 3e-4
optim = keras.optimizers.Adam(lr=base_lr)
model.compile(optimizer=optim,
              loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss)


def schedule(epoch, decay=0.9):
    return base_lr * decay**(epoch)

callbacks = [keras.callbacks.ModelCheckpoint('./checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                             verbose=1,
                                             save_weights_only=True),
             keras.callbacks.LearningRateScheduler(schedule)]

# 破片データ(テスト用)
#path = sorted(glob.glob('/Users/matsunagamasaaki/MasterResearch/fragment_annotation/data5/data/test_frames/*.jpg'))
# cupデータ(テスト用)
path = sorted(glob.glob('/Users/matsunagamasaaki/MasterResearch/cup_annotation/data1/data/test_data/*.jpg'))


# テスト用画像を数字順に並び替える関数
def sortedStringList(array=[]):
    sortDict = OrderedDict()
    for splitList in array:
        sortDict.update({splitList: [int(x) for x in re.split("(\d+)", splitList) if bool(re.match("\d*", x).group())]})
    return [sortObjKey for sortObjKey, sortObjValue in sorted(sortDict.items(), key=lambda x: x[1])]

#色の変換
def pil2cv(image):
    """ PIL型 -> OpenCV型 """
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image



# テスト用画像の呼び出しと推論
path_ = sortedStringList(path)
test5 = pickle.load(open('/Users/matsunagamasaaki/MasterResearch/ssd_keras/VOC_piece1.pkl', 'rb'))
keys = test5.keys()
ground_truth = test5.values()

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
            

            """
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
            """

            if label_name is "cupA":
                box_color = (0,255,0)
            elif "cupB":
                box_color = (0,255,255)


            display_txt = '{:0.2f}, {}'.format(score, label_name)
            coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
            # currentAxis.add_patch(plt.Rectangle(*coords,fill=False, edgecolor='white',linewidth=2))


            g_xmin=[int(round(i)) for i in gt_xmin * img.shape[1]]
            g_ymin=[int(round(i)) for i in gt_ymin * img.shape[0]]
            g_xmax=[int(round(i)) for i in gt_xmax * img.shape[1]]
            g_ymax=[int(round(i)) for i in gt_ymax * img.shape[0]]


            img = cv2.resize(img, (720, 405))
            imgg = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (85, 255, 0), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX

            ssd_center = (int(round(xmax - xmin + 1) / 2 + xmin), int(round(ymax - ymin + 1) / 2 + ymin))
            cv2.putText(img, label_name, ssd_center, font, 0.8, box_color, 2, cv2.LINE_AA)

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
            
            # クラスカテゴリの記述
            cv2.putText(img, str(y), (xmin, ymin - 2), font, 0.5, box_color, 1, cv2.LINE_AA)
            # FPS情報の記述
            cv2.putText(img, fps, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (85, 255, 0), 1)

            # 出力画像の色の変換
            imgg = pil2cv(imgg)

            # Gradcamの呼び出し
            gradcam = Gradcam(mode = 'dd')
            gradcam.conduct_gradcam(model,img)

            cv2.imshow('test', imgg)
            cv2.waitKey(1)





