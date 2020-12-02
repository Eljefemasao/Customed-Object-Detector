
import torch.nn as nn
import torch.nn.functional as F



class MarkNet(nn.Module):

    def __init__(self):
        super(MarkNet, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 =  nn.Linear(in_features=84, out_features=2)#(84, 10)

    def forward(self, x):
        x = self.max_pool2d(F.relu(self.conv1(x)))
        x = self.max_pool2d(F.relu(self.conv2(x)))
#         x = x.view(-1, 16*5*5)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



