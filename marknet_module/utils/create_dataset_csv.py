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
    label = 0
    if label_name == 'plateA':
        label = 0
    elif label_name == 'plateB':
        label = 1 
    elif label_name == 'plateC':
        label = 2
     
    image_label = [label for _ in range(len(image_path))]
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

    DATA_DIRECTORY = 'museumSquarePlate_annotation'

    # 訓練データまでのpath
    """
    data_path = ['/Users/matsunagamasaaki/MasterResearch/cup_annotation/data2/data/cupA/',
    '/Users/matsunagamasaaki/MasterResearch/cup_annotation/data2/data/cupB/',
    '/Users/matsunagamasaaki/MasterResearch/cup_annotation/data5/data/cupB/',
    '/Users/matsunagamasaaki/MasterResearch/cup_annotation/data6/data/cupA/',
    '/Users/matsunagamasaaki/MasterResearch/cup_annotation/data8/data/cupA/',
    '/Users/matsunagamasaaki/MasterResearch/cup_annotation/data8/data/cupB/',
    '/Users/matsunagamasaaki/MasterResearch/cup_annotation/data10/data/cupA/',
    '/Users/matsunagamasaaki/MasterResearch/cup_annotation/data10/data/cupB/',
    '/Users/matsunagamasaaki/MasterResearch/cup_annotation/data11/data/cupA/',
    '/Users/matsunagamasaaki/MasterResearch/cup_annotation/data11/data/cupB/',
    '/Users/matsunagamasaaki/MasterResearch/cup_annotation/data12/data/cupA/',
    '/Users/matsunagamasaaki/MasterResearch/cup_annotation/data12/data/cupB/']

    #['/Users/matsunagamasaaki/MasterResearch/cup_annotation/mark3/data/cupA/'
    ,'/Users/matsunagamasaaki/MasterResearch/cup_annotation/mark4/data/cupB/']

    # csvファイルの書き出し先
    output_path = '/Users/matsunagamasaaki/MasterResearch/cup_annotation/mark1/data/fullimage_data.csv'
    """

    """
    data_path=[
    '/Users/matsunagamasaaki/MasterResearch/museumPlate_annotation/data1/data/plateA/',
    '/Users/matsunagamasaaki/MasterResearch/museumPlate_annotation/data1/data/plateB/',
    '/Users/matsunagamasaaki/MasterResearch/museumPlate_annotation/data1/data/plateC/',    
    '/Users/matsunagamasaaki/MasterResearch/museumPlate_annotation/data2/data/plateA/',
    '/Users/matsunagamasaaki/MasterResearch/museumPlate_annotation/data2/data/plateB/',
    '/Users/matsunagamasaaki/MasterResearch/museumPlate_annotation/data2/data/plateC/',
    '/Users/matsunagamasaaki/MasterResearch/museumPlate_annotation/data3/data/plateA/',
    '/Users/matsunagamasaaki/MasterResearch/museumPlate_annotation/data3/data/plateB/'
    ,'/Users/matsunagamasaaki/MasterResearch/museumPlate_annotation/data3/data/plateC/']
    """


    data_path=[ '/Users/matsunagamasaaki/MasterResearch/'+DATA_DIRECTORY+'/data4/data/plateA/',
                '/Users/matsunagamasaaki/MasterResearch/'+DATA_DIRECTORY+'/data4/data/plateB/',
                '/Users/matsunagamasaaki/MasterResearch/'+DATA_DIRECTORY+'/data4/data/plateC/']

    output_path = '/Users/matsunagamasaaki/MasterResearch/'+DATA_DIRECTORY+'/data4/data/image_data.csv'


    """
    remote_data_path = ['/tmp/museumPlate_annotation/data2/data/plateA/',
                         '/tmp/museumPlate_annotation/data2/data/plateB/',
                         '/tmp/museumPlate_annotation/data2/data/plateC/',
                         '/tmp/museumPlate_annotation/data3/data/plateA/',
                         '/tmp/museumPlate_annotation/data3/data/plateB/',
                         '/tmp/museumPlate_annotation/data3/data/plateC/']
    """

    remote_data_path = ['/tmp/'+DATA_DIRECTORY+'/data4/data/plateA/',
                        '/tmp/'+DATA_DIRECTORY+'/data4/data/plateB/',
                        '/tmp/'+DATA_DIRECTORY+'/data4/data/plateC/']



    remote_output_path = '/tmp/'+DATA_DIRECTORY+'/data4/data/image_data.csv'

    # ローカル
    write_out_dataset_csv(data_path, output_path)

    # リモート
#    write_out_dataset_csv(remote_data_path, remote_output_path)
    











