import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import array_to_img, img_to_array, load_img, apply_affine_transform
import tensorflow as tf

import cv2 
import numpy as np
import pandas as pd


# MarkNet用のInputPipelineモジュール呼び出し
from marknet_keras_inputpipeline import BatchGenerator
# MarkNetモデルの呼び出し
from model.MarkNet_keras import Model_
# Gradcamモジュールの呼び出し
from utils.conduct_gradcam import Gradcam
#中間層の可視化モジュール呼び出し
#from utils.visualize_intermediate_weights_and_featuremaps import visualize_intermediate_activation_map

from utils.combine_multiple_loss import MultiLoss
from keras import metrics
from keras.callbacks import TensorBoard as tfboard
from keras.utils import plot_model
import matplotlib.pyplot as plt
from scikitplot.metrics import plot_confusion_matrix, plot_roc
import pickle
import ssl

from detect_specular_reflection import create_MS, convert_to_binary_img


ssl._create_default_https_context = ssl._create_unverified_context

# ローカル or リモートサーバー
position = True

# 学習データディレクトリ
DATA_DIRECTORY = 'museumPlate_annotation'
#DATA_DIRECTORY = 'CIFAR100'



if position:
    # 学習用データcsvファイル場所
    #DATAPATH='/Users/matsunagamasaaki/MasterResearch/'+DATA_DIRECTORY+'/data1/data/image_data.csv'
    # CIFAR100 [cup,plate,can,bottle]
    #DATAPATH = '/Users/matsunagamasaaki/MasterResearch/SSD_Keras/'+DATA_DIRECTORY+'/preprocessed_csv_data/train/image_data.csv'
    #'/Users/matsunagamasaaki/MasterResearch/cup_annotation/data1_mark/data/image_data.csv'
    #'/Users/matsunagamasaaki/MasterResearch/cup_annotation/mark1/data/data.csv'

    # Pot data
    #DATAPATH = '/Users/matsunagamasaaki/MasterResearch/annotation/original_data/annotation/preprocessed_csv_data/train/image_data.csv'

    # 検証用データcsv
    #VALDATAPATH = '/Users/matsunagamasaaki/MasterResearch/' + DATA_DIRECTORY + '/data2/data/image_data.csv'
    # CIFAR100 [cup,plate,can,bottle]
    #VALDATAPATH = '/Users/matsunagamasaaki/MasterResearch/SSD_Keras/'+DATA_DIRECTORY+'/preprocessed_csv_data/val/image_data.csv'
    #'/Users/matsunagamasaaki/MasterResearch/cup_annotation/mark1/data/val_data.csv'

    # Pot data
    #VALDATAPATH = '/Users/matsunagamasaaki/MasterResearch/annotation/original_data/annotation/preprocessed_csv_data/val/image_data.csv'

    # モデル評価用データcsv
    # plate
    #TESTDATAPATH = '/Users/matsunagamasaaki/MasterResearch/' + DATA_DIRECTORY + '/data2/data/image_data.csv'
    # CIFAR100
    #TESTDATAPATH = '/Users/matsunagamasaaki/MasterResearch/SSD_Keras/' + DATA_DIRECTORY + '/preprocessed_csv_data/test/image_data.csv'
    # Potデータ
    #TESTDATAPATH = '/Users/matsunagamasaaki/MasterResearch/annotation/original_data/annotation/preprocessed_csv_data/test/image_data.csv'

    # 複数カテゴリ
    DATAPATH = '/Users/matsunagamasaaki/MasterResearch/annotation/multi_category/train/image_data.csv'
    VALDATAPATH= '/Users/matsunagamasaaki/MasterResearch/annotation/multi_category/val/image_data.csv'
    TESTDATAPATH = '/Users/matsunagamasaaki/MasterResearch/annotation/multi_category/test/image_data.csv'


else:
    # 学習用データリモート用 plate
    #REMOTEDATAPATH='/tmp/'+DATA_DIRECTORY+'/data1/data/train_12000.csv'
    # CIFAR100 [cup,plate,can,bottle]
    #REMOTEDATAPATH = '/tmp/SSD_Keras/'+DATA_DIRECTORY+'/preprocessed_csv_data/train/image_data.csv'
    # CIFAR100 FULL DATA
    #REMOTEDATAPATH = '/tmp/SSD_Keras/' + DATA_DIRECTORY + '/preprocessed_csv_data/full_data/train/image_data.csv'
    # Potデータ
    #REMOTEDATAPATH = '/tmp/annotation_original/preprocessed_csv_data/train/image_data.csv'



    # 検証用データリモート用 plate
    #REMOTEVALDATAPATH= '/tmp/'+DATA_DIRECTORY+'/data1/data/val_3000.csv'
    # CIFAR100 [cup,plate,can,bottle]
    #REMOTEVALDATAPATH=  '/tmp/SSD_Keras/'+DATA_DIRECTORY+'/preprocessed_csv_data/val/image_data.csv'
    # CIFAR100 FULL DATA
    #REMOTEVALDATAPATH = '/tmp/SSD_Keras/' + DATA_DIRECTORY + '/preprocessed_csv_data/full_data/val/image_data.csv'
    # potデータ
    #REMOTEVALDATAPATH = '/tmp/annotation_original/preprocessed_csv_data/val/image_data.csv'


    # Potデータ
    #TESTDATAPATH = '/tmp/annotation_original/preprocessed_csv_data/test/image_data.csv'

    # 複数カテゴリ
    REMOTEDATAPATH = '/tmp/annotation/multi_category/train/image_data.csv'
    REMOTEVALDATAPATH = '/tmp/annotation/multi_category/val/image_data.csv'
    REMOTETESTDATAPATH = '/tmp/annotation/multi_category/test/image_data.csv'
    DATAPATH = REMOTEDATAPATH
    VALDATAPATH = REMOTEVALDATAPATH

FULLIMAGEDATAPATH='/Users/matsunagamasaaki/MasterResearch/cup_annotation/mark1/data/fullimage_data.csv'

best_model_path = './trained_weights/best_model.hdf5'
final_model_path = './trained_weights/final_model.hdf5'

image_shape = (300, 300, 3)
batch_size = 8#64

NUMCLASS=5


def recall(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (all_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_score(y_true, y_pred):
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def convert_keras_model_to_coreml_model(model):
    """
    kerasモデルをcoremlモデルに変換し保存する関数
    :param model: 学習済みkerasモデル
    :return:
    """
    import coremltools
    model.author = 'm.matsunaga'
    model.short_description = 'urasoe museum plate recognition'
    # model.input_description['image'] = 'take a input an image of a plate'
    # model.output_description['output'] = 'prediction of plate'
    output_labels = ['0', '1', '2', '3', '4']
    mlconverted_model = coremltools.converters.keras.convert(model)
    mlconverted_model.save('./trained_weights/best_model.mlmodel')

    return None


def plot_train_process(batch_level_history):

    #学習経過の可視化
    #metrics = ['loss', 'accuracy']  # 使用する評価関数を指定
    #metrics = [i for i in history.history.keys()]
    metric = ['mean_squared_error']
    print(metric)
    plt.figure(figsize=(10, 5))  # グラフを表示するスペースを用意

    #for i in range(len(metrics)):
    #metric = metrics[i]
    #plt.subplot(1, 2, i+1)  # figureを1×2のスペースに分け、i+1番目のスペースを使う
    plt.title(metric,size=15)  # グラフのタイトルを表示

    #plt_train = history.history[metric]  # historyから訓練データの評価を取り出す
    #plt_test = history.history['val_' + metric]  # historyからテストデータの評価を取り出す

    plt_train = batch_level_history.mse
    plt_test = batch_level_history.val_mse

    # plotデータの一時保存
    pd.to_pickle(plt_train, '../CIFAR100/plot_pkl_data/inception/train.pkl')
    pd.to_pickle(plt_test, '../CIFAR100/plot_pkl_data/inception/val.pkl')

    tmp1 = len(plt_train) / 160
    tmp2 = len(plt_train) / 40
    x1 = np.arange(0, tmp1,  0.00625)
    x2 = np.arange(0, tmp2, 0.025)

    plt.plot(x1, plt_train, label='train_mse', color='magenta')  # 訓練データの評価をグラフにプロット
    plt.plot(x2, plt_test, label='test_mse', color='darkmagenta')  # テストデータの評価をグラフにプロット

    plt.legend()  # ラベルの表示
    plt.ylabel('mean square error',size=15)
    plt.xlabel('iteration epoch', size=15)
    plt.xticks(fontsize=12, rotation='vertical')
    plt.yticks(fontsize=12)
    plt.grid()

    plt.savefig('./train_process_fig.png')





def prepare_multiple_category_outputs(model , i, gradcam):
    """
    Plate_A, Plate_B, Plate_Cの画像を同時にGradcamで可視化するため、入力画像を用意する関数
    :param model: 学習済みkerasモデル
    :param i: インデックス
    :param gradcam: Gradcamモジュール
    :return: None
    """
    def concat_vh(list_2d):
        # return final image
        return cv2.vconcat([cv2.hconcat(list_h)
                            for list_h in list_2d])

    categories = ['a', 'b', 'c']
    horizontal_imgs = []

    for category in categories:
        # plateデータの可視化
        #path = '/Users/matsunagamasaaki/MasterResearch/'+DATA_DIRECTORY+'/test_csvs/image_data_'+category+'.csv'
        # CIFAR100用の可視化
        path = TESTDATAPATH
        df = pd.read_csv(path)
        df_sorted = df.sample(frac=1)
        #df_sorted = df.sort_values(by='image', ascending=False)
        x_test = df_sorted['image'].tolist()
        y_test = [list(np.eye(1, M=NUMCLASS, k=i, dtype=np.int8)[0]) for i in df_sorted['label'].tolist()]

        # テスト画像
        img = load_img(x_test[i])
        print("path", x_test[i])
        img = tf.keras.preprocessing.image.img_to_array(img)
        # numpy array to tensor に変換
        tensor_img = tf.convert_to_tensor(img, dtype=tf.float32)
        tensor_img = tf.image.resize_with_crop_or_pad(image=tensor_img, target_height=image_shape[0],
                                                      target_width=image_shape[1])
        with tf.Session() as sess:
            img = sess.run(tensor_img)
        expanded_img = np.expand_dims(img, 0)

        # モデルの推論
        output = model.predict(expanded_img, batch_size=1, verbose=1)

        # 中間層のアクティベーション可視化
        #intermediate_img = visualize_intermediate_activation_map(model=model, url=x_test[i])

        # Gradcam オリジナル画像可視化用
        img = cv2.imread(x_test[i])
        im_h = gradcam.conduct_gradcam(model, img, image_shape=image_shape, category=category)
        horizontal_imgs.append(im_h)
        print('model:', output)
        print('GL:', y_test[i])
        print('===============')

    """
    # 入力画像中のspecularの特定
    specular_img = create_MS(img=img, fmap=img)
    specular_img = cv2.resize(specular_img, (300,300))
    # O/1に変換
    specular_img_ = convert_to_binary_img(specular_img)
    a = cv2.hconcat([specular_img_[:,:,0],specular_img_[:,:,1],specular_img_[:,:,2],specular_img_[:,:,3],specular_img_[:,:,4]])
    b = intermediate_img#[:300, :300]
    c=b[:300, :300] * a[:300, :300]
    cv2.imshow('Attenuation', c)
    cv2.imshow('specular', cv2.resize(specular_img, (300,300)))
    cv2.imshow('activations', intermediate_img)
    """

    im_h = concat_vh([[horizontal_imgs[0]],
                      [horizontal_imgs[1]],
                      [horizontal_imgs[2]]])
    cv2.imshow('Attention/ Class Activation Map', im_h)
    #cv2.imwrite('/Users/matsunagamasaaki/Documents/ipsj_v4/UTF8/image/attention_square.png',im_h)
    return None


def main():

    # MarkNetの呼び出し
    net = Model_(input_shape=image_shape)
    #model = net.MarkNet()  # BaseNet
    #model = net.NormalNet()
    #model = net.AttentionMarkNet(transfar=False, inception=False, vgg=True)
    model = net.WhiteAttenuation_And_AttentionMarkNet()
    #model = net.Inceptionv3(transfar=True)
    #model = net.VGG(transfar=True)
    #model = net.ResNet()

    # モデル図の可視化
    plot_model(model, to_file='model.png', show_shapes=True)

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    #opt= keras.optimizers.Adam(lr=0.0001)
    # Let's train the model using RMSprop
    model.compile(loss=MultiLoss(NUMCLASS).compute_loss,#'categorical_crossentropy',##,
              optimizer=opt,
              metrics=[keras.metrics.categorical_accuracy, precision, recall, keras.metrics.mean_squared_error])


    df_train = pd.read_csv(DATAPATH)
    df_valid = pd.read_csv(VALDATAPATH)


    # prepare train and test data sets (one-hotベクターに変換
    x_train = df_train['image'].tolist()
    y_train = [ list(np.eye(1, M=NUMCLASS, k=int(i), dtype=np.int8)[0]) for i in df_train['label'].tolist()]
    x_val = df_valid['image'].tolist()
    y_val = [ list(np.eye(1, M=NUMCLASS, k=int(i), dtype=np.int8)[0]) for i in df_valid['label'].tolist()]

  
    # ジェネレーターの呼び出し
    train_batch_generator = BatchGenerator(x_train, y_train, image_shape, batch_size)
    test_batch_generator = BatchGenerator(x_val, y_val, image_shape, batch_size)


    # start training
    chk_point = keras.callbacks.ModelCheckpoint(filepath = best_model_path, monitor='val_loss',
                                                verbose=1, save_best_only=True, save_weights_only=True,
                                                mode='min', period=1)
    # tensor boardへの書き出し
    tfboard_cb = tfboard(log_dir='./tensorboard_log', histogram_freq=0, batch_size=batch_size, write_graph=True,
                          write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                         embeddings_metadata=None)

    class BatchHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.val_losses = []
            self.mse = []
            self.val_mse = []

        def on_batch_end(self, batch=128, logs={}):
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            self.mse.append(logs.get('mean_squared_error'))
            self.val_mse.append(logs.get('val_mean_squared_error'))

    """
    batch_level_history = BatchHistory()
    history = model.fit_generator(train_batch_generator, epochs=50,
                                    steps_per_epoch=train_batch_generator.batches_per_epoch,
                                    verbose=1,
                                    validation_data=test_batch_generator,
                                    validation_steps=test_batch_generator.batches_per_epoch,
                                    shuffle=True,
                                    callbacks=[chk_point, tfboard_cb, batch_level_history])

    model.save(final_model_path)
    #model.save_weights('marknet_finalconv.hdf5', save_weights_only=True)

    # kerasモデルをcoremlモデルに変換して保存する
    #convert_keras_model_to_coreml_model(model=model)


    # 学習経過のグラフ化
    plot_train_process(batch_level_history=batch_level_history)

    """
    ## ORIGINAL DATA

    # モデルの学習済み重み呼び出し
    model.load_weights('./trained_weights/best_model.hdf5', by_name=True)
    # BaseNet for plateマスク
    # model.load_weights('./trained_weights/best_model_plate.hdf5', by_name=True)
    # attention モデル for square plate
    #model.load_weights('./trained_weights/attention_best_model_for_squarePlate.hdf5')
    # attenuation モデル for plate
    #model.load_weights('./trained_weights/attenuation_and_attention_best_model_for_plate.hdf5',by_name=True)
    #model.load_weights('./trained_weights/Base+ABN+Ours_plate_x1.hdf5', by_name=True)

    ## CIFAR100 DATA

    # VGG16
    #model.load_weights('./trained_weights/VGG_cifar100x50.hdf5', by_name=True)
    # InceptionV3
    #model.load_weights('./trained_weights/Inceptionv3_cifar100x5.hdf5', by_name=True)

    # Normal+ABN
    #model.load_weights('./trained_weights/normal+ABN_cifar100.hdf5', by_name=True)
    # Normal+ABN+Mark
    #model.load_weights('./trained_weights/normal+ABN+Mark_cifar100.hdf5', by_name=True)

    # VGG+ABN
    #model.load_weights('./trained_weights/VGG+ABN_cifar100x50.hdf5', by_name=True)
    # VGG+ABN+Mark
    #model.load_weights('./trained_weights/VGG+ABN+Mark_cifar100x50.hdf5', by_name=True)

    # Inception+ABN
    #model.load_weights('./trained_weights/Inceptionv3+ABN_cifar100x50.hdf5', by_name=True)
    # Inception+ABN+Mark
    #model.load_weights('./trained_weights/Inceptionv3+ABN+Mark_cifar100x50.hdf5', by_name=True)


    ## Potデータ
    #model.load_weights('./trained_weights/normal_pot_x1.hdf5', by_name=True)
    #model.load_weights('./trained_weights/normal+ABN+Mark_pot_x1.hdf5', by_name=True)
    #model.load_weights('./trained_weights/vgg_pot_x1.hdf5', by_name=True)
    #model.load_weights('./trained_weights/vgg_pot_x50_category_100.hdf5', by_name=True)
    #model.load_weights('./trained_weights/normal+ABN+Mark_pot_x50_category_100.hdf5', by_name=True)
    #model.load_weights('./trained_weights/normal+ABN_pot_x50_category_100.hdf5', by_name=True)
    #model.load_weights('./trained_weights/VGG+ABN+Mark_pot_x50_category_100.hdf5', by_name=True)



    # モデルの構造サマリ
    model.summary()

    # モデルの構造プロット
    keras.utils.plot_model(model, "./model_structure_image/model.png", show_shapes=True)


    error = []
    accu = []
    reca = []
    import statistics
    #for _ in range(10):

    # 評価用テストデータ
    df_test = pd.read_csv(TESTDATAPATH)
    df_test = df_test.sample(frac=1)
    x_test = df_test['image'].tolist()
    y_test = [ list(np.eye(1, M=NUMCLASS, k=int(i), dtype=np.int8)[0]) for i in df_test['label'].tolist()]

    x_test_ = []
    for path in x_test:
        img = load_img(path, target_size=image_shape)
        img = tf.keras.preprocessing.image.img_to_array(img)
        x_test_.append(img)

    x_test_ = np.array(x_test_)/255
    y_test_ = np.array(y_test)

    # モデルの評価
    evaluate = True
    if evaluate:
        scores = model.evaluate(x_test_, y_test_, verbose=1, batch_size=32)
        print('tmp:', scores)
        error.append(float(scores[-1]))
        accu.append(float(scores[1]))
        reca.append(float(scores[-2]))
        print("error", statistics.median(error))
        print("accuracy", statistics.median(accu))
        print("recall", statistics.median(reca))

    # Gradcamモジュールの呼び出し
    gradcam = Gradcam(mode = 'dd')


    for i in range(1000):
        prepare_multiple_category_outputs(model=model, i=i, gradcam=gradcam)
        #cv2.waitKey(3000)
        cv2.waitKey(200)



if __name__ == "__main__":
    main()





























