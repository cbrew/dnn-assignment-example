# model.py
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class HyperParams:
    input_dim: int = 784
    num_hid1_units: int = 300
    num_hid2_units: int = 100
    num_classes: int = 10
    lr: float = 0.01
    num_epochs: int = 20
    batch_size: int = 64
    use_dropout: bool = False
    dropout_rate: float = 0.25


class DNN(nn.Module):
    def __init__(self, config: HyperParams):
        super().__init__()

        self.config = config

        if self.config.use_dropout:
            self.fc1 = nn.Linear(config.input_dim, config.num_hid1_units)
            self.dp1 = nn.Dropout(p=config.dropout_rate)
            self.fc2 = nn.Linear(config.num_hid1_units, config.num_hid2_units)
            self.dp2 = nn.Dropout(p=config.dropout_rate)
            self.fc3 = nn.Linear(config.num_hid2_units, config.num_classes)
        else:
            self.fc1 = nn.Linear(config.input_dim, config.num_hid1_units)
            self.fc2 = nn.Linear(config.num_hid1_units, config.num_hid2_units)
            self.fc3 = nn.Linear(config.num_hid2_units, config.num_classes)

        # Initialization logic
        stddev1 = 2 / np.sqrt(config.input_dim + config.num_hid1_units)
        nn.init.normal_(self.fc1.weight, std=stddev1)
        nn.init.zeros_(self.fc1.bias)

        stddev2 = 2 / np.sqrt(config.num_hid1_units + config.num_hid2_units)
        nn.init.normal_(self.fc2.weight, std=stddev2)
        nn.init.zeros_(self.fc2.bias)

        stddev3 = 2 / np.sqrt(config.num_hid2_units + config.num_classes)
        nn.init.normal_(self.fc3.weight, std=stddev3)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        # x: (batch_size, 784)
        if self.config.use_dropout:
            h1 = F.relu(self.dp1(self.fc1(x)))
            h2 = F.relu(self.dp2(self.fc2(h1)))
            out = self.fc3(h2)
        else:
            h1 = F.relu(self.fc1(x))
            h2 = F.relu(self.fc2(h1))
            out = self.fc3(h2)
        return out
