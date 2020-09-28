#
# MarkNetモデルの学習に利用する画像path+ラベルを集約したcsvファイル
#
import pandas as pd
import glob 


def get_imagepath_and_label_df(data_path):
    """訓練・検証用画像データのpathを取得し、ラベルのannotationを行ってdfを生成
    """    
    #　画像データのpath
    image_path = glob.glob(data_path+'*.jpg')
    # 画像データのラベル
    label_name = data_path.split('/')[-2]

    image_label = [label_name for _ in range(len(image_path))]
    df = pd.DataFrame({'image':image_path, 'label':image_label})
    return df

def write_out_dataset_csv(data_path, output_path):
    """dfのconcatおよびcsv形式で書き出し
    """
    # pd.concat用
    df_list = []
    for path in data_path:
        df = get_imagepath_and_label_df(path)
        df_list.append(df)

    if len(df_list) > 1:
        final_df = pd.concat(df_list,axis=0)
    else:
        final_df = df_list[0]

    final_df.to_csv(output_path)



if __name__ == '__main__':

    # 訓練データまでのpath
    data_path = ['/Users/matsunagamasaaki/MasterResearch/cup_annotation/mark1/data/cupA/'\
        ,'/Users/matsunagamasaaki/MasterResearch/cup_annotation/mark2/data/cupB/']
    # csvファイルの書き出し先
    output_path = '/Users/matsunagamasaaki/MasterResearch/cup_annotation/mark1/data/data.csv'

    write_out_dataset_csv(data_path, output_path)
    











