


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from MarkNet import MarkNet



marknet = MarkNet()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)






