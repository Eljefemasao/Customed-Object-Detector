


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model.MarkNet import MarkNet
from utils.data_loader import DatasetLoader
import cv2

from torchvision import transforms


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# 学習・検証用データcsvファイル場所
DATAPATH='/Users/matsunagamasaaki/MasterResearch/cup_annotation/mark1/data/data.csv'

# 学習済みモデルの保管ディレクトリ
MODELPATH='/Users/matsunagamasaaki/MasterResearch/ssd_keras/marknet_module/trained_model'



# GPUの利用があるのか確認
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

dataset_loader = DatasetLoader(csv_file_path=DATAPATH,transform=transform)

# モデルの呼び出し
marknet = MarkNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(marknet.parameters(), lr=0.001, momentum=0.9)



for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataset_loader, 0):
        # get the inputs; data is a list of [inputs, labels]

        inputs = data[i]
        labels = data[i]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = marknet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


# モデルの書き出し
torch.save(marknet.state_dict(), MODELPATH)



