
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_plot():

    
    plt.figure(figsize=(10,5))
    plt_train = pd.read_pickle('./plot_pkl_data/vgg/train.pkl')
    plt_val = pd.read_pickle('./plot_pkl_data/vgg/val.pkl')
    print(plt_val)
    
    plt.title('vgg_error',size=15)
    
    
    tmp1 = len(plt_train) / 160
    tmp2 = len(plt_val) / 40
    x1 = np.arange(0, tmp1,  0.00625)
    x2 = np.arange(0, tmp2, 0.025)

#    plt.plot(x1, plt_train, label='train_mse', color='magenta')  # 訓練データの評価をグラフにプロット
    plt.plot(x2, plt_val, label='test_mse', color='darkmagenta')  # テストデータの評価をグラフにプロット

    plt.legend()  # ラベルの表示
    plt.ylabel('mean square error',size=15)
    plt.xlabel('iteration epoch', size=15)
    plt.xticks(fontsize=12, rotation='vertical')
    plt.yticks(fontsize=12)
    plt.grid()
    plt.show()

    
    return None


if __name__ == '__main__':
    get_plot()
