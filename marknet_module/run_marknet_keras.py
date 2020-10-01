
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
from marknet_keras_inputpipeline import CSVSequence, BatchGenerator

# MarkNetモデルの呼び出し
from model.MarkNet_keras import Model

# Gradcamモジュールの呼び出し
from utils.conduct_gradcam import Gradcam


# 学習・検証用データcsvファイル場所
DATAPATH='/Users/matsunagamasaaki/MasterResearch/cup_annotation/mark1/data/data.csv'
VALDATAPATH='/Users/matsunagamasaaki/MasterResearch/cup_annotation/mark1/data/val_data.csv'

FULLIMAGEDATAPATH='/Users/matsunagamasaaki/MasterResearch/cup_annotation/mark1/data/fullimage_data.csv'

best_model_path = './trained_weights/best_model.hdf5'
final_model_path = './trained_weights/final_model.hdf5'

image_shape = (64, 64, 3)
batch_size = 64


def main():

    # MarkNetの呼び出し
    net=Model()
    model=net.MarkNet()
   
# initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

    df_train = pd.read_csv(DATAPATH)
    df_valid = pd.read_csv(VALDATAPATH)

    # マークではなく、cupの全貌を収めた画像(検証用)
    df_fullimage = pd.read_csv(FULLIMAGEDATAPATH)
    x_train, x_valid, y_train, y_valid = \
            train_test_split(df_fullimage['image'].tolist(), df_fullimage['label'].tolist(), test_size=0.2, random_state=42)
    
    df_train = pd.DataFrame({'image':x_train,'label':y_train})
    df_valid = pd.DataFrame({'image':x_valid,'label':y_valid})

    # prepare train and test data sets
    x_train = df_train['image'].tolist()
    y_train = [ list(np.eye(1, M=2, k=i, dtype=np.int8)[0]) for i in df_train['label'].tolist()]
    x_test = df_valid['image'].tolist()
    y_test = [ list(np.eye(1, M=2, k=i, dtype=np.int8)[0]) for i in df_valid['label'].tolist()]

  
    # ジェネレーターの呼び出し
    train_batch_generator = BatchGenerator(x_train, y_train, image_shape, batch_size)
    test_batch_generator = BatchGenerator(x_test, y_test, image_shape, batch_size)

    # start training
    chk_point = keras.callbacks.ModelCheckpoint(filepath = best_model_path, monitor='val_loss',
                                                verbose=1, save_best_only=True, save_weights_only=True,
                                                mode='min', period=1)

    """
    fit_history = model.fit_generator(train_batch_generator, epochs=15,
                                    steps_per_epoch=train_batch_generator.batches_per_epoch,
                                    verbose=1,
                                    validation_data=test_batch_generator,
                                    validation_steps=test_batch_generator.batches_per_epoch,
                                    shuffle=True,
                                    callbacks=[chk_point])
    model.save(final_model_path)
    #model.save_weights('marknet_finalconv.hdf5', save_weights_only=True)
    """

    # モデルの学習済み重み呼び出し
    model.load_weights('./trained_weights/best_model.hdf5')

    # モデルの構造サマリ
    model.summary()
    # モデルの構造プロット
    keras.utils.plot_model(model, "./model_structure_image/model.png", show_shapes=True)

    # Gradcamモジュールの呼び出し
    gradcam = Gradcam(mode = 'dd')

    for i in range(1000):
        # テスト画像
        img = load_img(x_test[i], target_size=(64,64))
        img = tf.keras.preprocessing.image.img_to_array(img)

        # 入力画像の回転-90d
      #  img = apply_affine_transform(img, channel_axis=2, theta=-90, fill_mode="nearest", cval=0.)
        expanded_img = np.expand_dims(img, 0)

        output = model.predict(expanded_img, batch_size=1, verbose=1)
        
        gradcam.conduct_gradcam(model,img)
        print('model:',output)
        print('GL:', y_test[i])
        print('===============')
        cv2.waitKey(1)




    




if __name__ == "__main__":
    main()



























