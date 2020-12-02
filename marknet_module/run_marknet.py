


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model.MarkNet import MarkNet
from utils.data_loader import DatasetLoader
import cv2

from torchvision import transforms
import torchvision

from tqdm import tqdm 

transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(1.0, 1.0), ratio=(1.0, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# 学習・検証用データcsvファイル場所
DATAPATH='/Users/matsunagamasaaki/MasterResearch/cup_annotation/mark1/data/data.csv'

# 学習済みモデルの保管ディレクトリ
MODELPATH='/Users/matsunagamasaaki/MasterResearch/ssd_keras/marknet_module/trained_model/mode.pt'



# GPUの利用があるのか確認
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

dataset_loader = \
    DatasetLoader(csv_file_path=DATAPATH,transform=transform)

trainloader = torch.utils.data.DataLoader(dataset_loader, batch_size = 6, shuffle = True, drop_last=True, num_workers=0)

# モデルの呼び出し
marknet = MarkNet()


criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(marknet.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(marknet.parameters(), lr=0.05)


marknet.train()
# 学習結果の保存用
history = {
    'train_loss': [],
    'test_loss': [],
    'test_acc': [],
}

Epoch = 30 
for epoch in range(Epoch):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader, 0)):

        if i > len(trainloader):
            break

        # get the inputs; data is a list of [inputs, labels]
        inputs = data[0]
        labels = data[1]
        # 一度計算された勾配を0にリセット
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = marknet(inputs)
        loss = criterion(outputs, labels)

        # 誤差のbackpropagation
        loss.backward()

        # backpropagationの値による重みの更新
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print('Epoch: [%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            
            running_loss = 0.0
            history['train_loss'].append(loss)

print('Finished Training')


# モデルの書き出し
torch.save(marknet.state_dict(), MODELPATH)



