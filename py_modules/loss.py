import torch.nn as nn
import torch
import numpy as np

abundances = [0.16327756, 
              0.09594557, 
              0.04495886, 
              0.16392787,
              0.28916235, 
              0.21721568, 
              0.01044246, 
              0.01506966]
weights_balanced = (np.mean(abundances) / abundances).round(2)

class ce_balanced(nn.CrossEntropyLoss):
    def __init__(self):
        super(ce_balanced, self).__init__()
        self.weight = torch.FloatTensor(weights_balanced)