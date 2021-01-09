import pandas as pd
import cv2
import numpy as np



def cifar100_data_pipline():
    """ CIFAR100 datasetより"bottle","can","plate","cup"
        TODO:
              bowls,
    """
    df_train = pd.read_csv('./data/cifar100_train.csv')
    df_test = pd.read_csv('./data/cifar100_test.csv')

    df = df_train

    labels = df['label'].tolist()
    print(pd.unique(labels))
    labels = ['cup','can','bottle','plate']

    df_train_ = pd.DataFrame({'image':[], 'label':[]})
    df_test_ = pd.DataFrame({'image':[], 'label':[]})    

    for idx, df in enumerate([df_train, df_test]):
        for label in labels:            
            df_data=df[df['image'].where(df['label'] == label).notna()]
            if idx == 0:
                df_train_ = pd.concat([df_train_, df_data], axis=0)
            else:
                df_test_ = pd.concat([df_test_, df_data], axis=0)

    # インデックスの整頓
    df_train_ = df_train_.reset_index(drop=True)
    df_test_ = df_test_.reset_index(drop=True)


    image_train = df_train_['image'].tolist()
    label_train = df_train_['label'].tolist()
    image_test = df_test_['image'].tolist()
    label_test = df_test_['label'].tolist()
    label_train_ = []
    label_test_ = []

    
    print(image_train)
    labels = ['cup','can','bottle','plate']

    for idx, label_list in enumerate([label_train, label_test]):
        if idx == 0:
            tmp = label_train_
        else:
            tmp = label_test_

        for label in label_list:
            if label == labels[0]:
                tmp.append(0)
            elif label == labels[1]:
                tmp.append(1)
            elif label == labels[2]:
                tmp.append(2)
            elif label == labels[3]:
                tmp.append(3)

    df_train_result = pd.DataFrame({'image': image_train, 'label':label_train_})
    df_test_result = pd.DataFrame({'image':image_test, 'label':label_test_})

    # val dataframeの作成
    shuffled_train_df = df_train_result.sample(frac=1, random_state=0)
    # 学習データの乱数分割
    df_train_result = shuffled_train_df[:1600]
    # 検証用データの乱数分割
    df_val_result = shuffled_train_df[1600:]

    print('train',df_train_result)
    print('val', df_val_result)
    print('test',df_test_result)

    
    # csvファイルの書き出し
    df_train_result.to_csv('./preprocessed_csv_data/train/image_data.csv')
    df_val_result.to_csv('./preprocessed_csv_data/val/image_data.csv')
    df_test_result.to_csv('./preprocessed_csv_data/test/image_data.csv')


    """
    画像データの可視化
    data_paths=df_data['image'].tolist()
    for path in data_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (300,300))
        cv2.imshow('bottole',img)
        cv2.waitKey(300)
    cv2.destroyAllwindows()

    """
    
    return df_train_, df_test_


def cifar100_full_data_pipline():
    
    """ CIFAR100 datasetより全てのカテゴリを含むデータセットの構築
    """
    
    df_train = pd.read_csv('./data/cifar100_train.csv')
    df_test = pd.read_csv('./data/cifar100_test.csv')

    df = df_train
    print(df)
    print(df_test)
    labels = df['label'].tolist()
    print('all_index:',pd.unique(labels).tolist())

    labels = pd.unique(labels).tolist()

    df_train_ = pd.DataFrame({'image':[], 'label':[]})
    df_test_ = pd.DataFrame({'image':[], 'label':[]})    

    for idx, df in enumerate([df_train, df_test]):
        for label in labels:
            df_data=df[df['image'].where(df['label'] == label).notna()]
            if idx == 0:
                df_train_ = pd.concat([df_train_, df_data], axis=0)
            else:
                df_test_ = pd.concat([df_test_, df_data], axis=0)

    # インデックスの整頓
    df_train_ = df_train_.reset_index(drop=True)
    df_test_ = df_test_.reset_index(drop=True)


    image_train = df_train_['image'].tolist()
    label_train = df_train_['label'].tolist()
    image_test = df_test_['image'].tolist()
    label_test = df_test_['label'].tolist()
    label_train_ = []
    label_test_ = []

    for idx, label_list in enumerate([label_train, label_test]):
        if idx == 0:
            tmp = label_train_
        else:
            tmp = label_test_

        for label in label_list:
            tmp.append(labels.index(label))

    df_train_result = pd.DataFrame({'image': image_train, 'label':label_train_})
    df_test_result = pd.DataFrame({'image':image_test, 'label':label_test_})

    # val dataframeの作成
    shuffled_train_df = df_train_result.sample(frac=1, random_state=0)
    # 学習データの乱数分割
    df_train_result = shuffled_train_df[:40000]
    # 検証用データの乱数分割
    df_val_result = shuffled_train_df[40000:]

    print('train',df_train_result)
    print('val', df_val_result)
    print('test',df_test_result)

    
    # csvファイルの書き出し
    df_train_result.to_csv('./preprocessed_csv_data/full_data/train/image_data.csv')
    df_val_result.to_csv('./preprocessed_csv_data/full_data/val/image_data.csv')
    df_test_result.to_csv('./preprocessed_csv_data/full_data/test/image_data.csv')

    """
    画像データの可視化
    data_paths=df_data['image'].tolist()
    for path in data_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (300,300))
        cv2.imshow('bottole',img)
        cv2.waitKey(300)
    cv2.destroyAllwindows()
    """
    
    return df_train_, df_test_


if __name__ == '__main__':
    # cup, plate, can, bottleのデータ用意
#    train, test = cifar100_data_pipline()
    # 全カテゴリのデータ用意
    cifar100_full_data_pipline()

