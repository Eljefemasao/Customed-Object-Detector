import keras 
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

from utils.combine_multiple_loss import MultiLoss
from keras import metrics
from keras.callbacks import TensorBoard as tfboard
from keras.utils import plot_model


# ローカル or リモートサーバー
position = True

# 学習データディレクトリ
DATA_DIRECTORY = 'museumPlate_annotation'

if position:
    # 学習用データcsvファイル場所
    DATAPATH='/Users/matsunagamasaaki/MasterResearch/'+DATA_DIRECTORY+'/data1/data/image_data.csv'
    #'/Users/matsunagamasaaki/MasterResearch/cup_annotation/data1_mark/data/image_data.csv'
    #'/Users/matsunagamasaaki/MasterResearch/cup_annotation/mark1/data/data.csv'

    # 検証用データcsv
    VALDATAPATH= '/Users/matsunagamasaaki/MasterResearch/'+DATA_DIRECTORY+'/data2/data/image_data.csv'
    #'/Users/matsunagamasaaki/MasterResearch/cup_annotation/mark1/data/val_data.csv'

else:
    # 学習用データリモート用
    REMOTEDATAPATH='/tmp/'+DATA_DIRECTORY+'/data1/data/image_data.csv'
    DATAPATH = REMOTEDATAPATH


    # 検証用データリモート用
    REMOTEVALDATAPATH= '/tmp/'+DATA_DIRECTORY+'/data2/data/image_data.csv'
    VALDATAPATH = REMOTEVALDATAPATH


FULLIMAGEDATAPATH='/Users/matsunagamasaaki/MasterResearch/cup_annotation/mark1/data/fullimage_data.csv'

best_model_path = './trained_weights/best_model.hdf5'
final_model_path = './trained_weights/final_model.hdf5'

image_shape = (300, 300, 3)
batch_size = 32#64

NUMCLASS=5

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

def prepare_multiple_category_outputs(model , i, gradcam):
    """
    Plate_A, Plate_B, Plate_Cの画像を同時にGradcamで可視化するため、入力画像を用意する関数
    :param model: 学習済みkerasモデル
    :param i: インデックス
    :param gradcam: Gradcamモジュール
    :return: None
    """

    categories = ['a', 'b', 'c']
    for category in categories:
        path = '/Users/matsunagamasaaki/MasterResearch/'+DATA_DIRECTORY+'/test_csvs/image_data_'+category+'.csv'
        df = pd.read_csv(path)
        df_sorted = df.sort_values(by='image', ascending=False)
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
        output = model.predict(expanded_img, batch_size=1, verbose=1)

        # Gradcam オリジナル画像可視化用
        img = cv2.imread(x_test[i])
        gradcam.conduct_gradcam(model, img, image_shape=image_shape, category=category)
        print('model:', output)
        print('GL:', y_test[i])
        print('===============')

    return None

def main():

    # MarkNetの呼び出し
    net = Model_(input_shape=image_shape)
    #model = net.MarkNet()
    #model = net.AttentionMarkNet()
    model = net.WhiteAttenuation_And_AttentionMarkNet()

    # モデル図の可視化
    plot_model(model, to_file='model.png', show_shapes=True)

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    #opt= keras.optimizers.Adam(lr=0.0001)
    # Let's train the model using RMSprop
    model.compile(loss=MultiLoss(NUMCLASS).compute_loss,
              optimizer=opt,
              metrics=[metrics.categorical_accuracy])

    df_train = pd.read_csv(DATAPATH)
    df_valid = pd.read_csv(VALDATAPATH)

    # マークではなく、cupの全貌を収めた画像(検証用)
#    df_fullimage = pd.read_csv(FULLIMAGEDATAPATH)
    x_train, x_valid, y_train, y_valid = \
            train_test_split(df_train['image'].tolist(), df_train['label'].tolist(), test_size=0.2, random_state=42)

    x_, x_valid, y_, y_valid = \
            train_test_split(df_valid['image'].tolist(), df_valid['label'].tolist(), test_size=0.2, random_state=42)
    
    df_train = pd.DataFrame({'image':x_train,'label':y_train})
    df_valid = pd.DataFrame({'image':x_valid,'label':y_valid})

    # 検証用に画像番号順に並び替え
#    df_valid=df_valid.sort_values(by='image',ascending=False)


    # prepare train and test data sets (one-hotベクターに変換
    x_train = df_train['image'].tolist()
    y_train = [ list(np.eye(1, M=NUMCLASS, k=i, dtype=np.int8)[0]) for i in df_train['label'].tolist()]
    x_test = df_valid['image'].tolist()
    y_test = [ list(np.eye(1, M=NUMCLASS, k=i, dtype=np.int8)[0]) for i in df_valid['label'].tolist()]

  
    # ジェネレーターの呼び出し
    train_batch_generator = BatchGenerator(x_train, y_train, image_shape, batch_size)
    test_batch_generator = BatchGenerator(x_test, y_test, image_shape, batch_size)


    # start training
    chk_point = keras.callbacks.ModelCheckpoint(filepath = best_model_path, monitor='val_loss',
                                                verbose=1, save_best_only=True, save_weights_only=True,
                                                mode='min', period=1)
    # tensor boardへの書き出し
    tfboard_cb = tfboard(log_dir='./tensorboard_log', histogram_freq=0, batch_size=32, write_graph=True,
                         write_grads=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                         embeddings_metadata=None)

    """
    fit_history = model.fit_generator(train_batch_generator, epochs=1,
                                    steps_per_epoch=train_batch_generator.batches_per_epoch,
                                    verbose=1,
                                    validation_data=test_batch_generator,
                                    validation_steps=test_batch_generator.batches_per_epoch,
                                    shuffle=True,
                                    callbacks=[chk_point, tfboard_cb])

    model.save(final_model_path)
    #model.save_weights('marknet_finalconv.hdf5', save_weights_only=True)

    # kerasモデルをcoremlモデルに変換して保存する
    convert_keras_model_to_coreml_model(model=model)
    """

    # モデルの学習済み重み呼び出し
    #model.load_weights('./trained_weights/best_model.hdf5')
    # attention モデル for square plate
    #model.load_weights('./trained_weights/attention_best_model_for_squarePlate.hdf5')
    # attenuation モデル for plate
    model.load_weights('./trained_weights/attenuation_and_attention_best_model_for_plate.hdf5')

    # モデルの構造サマリ
    model.summary()
    # モデルの構造プロット
    keras.utils.plot_model(model, "./model_structure_image/model.png", show_shapes=True)

    # Gradcamモジュールの呼び出し
    gradcam = Gradcam(mode = 'dd')


    for i in range(1000):
        prepare_multiple_category_outputs(model=model, i=i, gradcam=gradcam)
        cv2.waitKey(3000)
        #cv2.waitKey(30)



if __name__ == "__main__":
    main()



























