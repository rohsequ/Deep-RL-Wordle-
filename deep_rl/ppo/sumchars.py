from typing import List

import numpy as np
import torch
from torch import nn
from wordle.const import WORDLE_CHARS, WORDLE_N

class SumChars(nn.Module):
    def __init__(self, obs_size: int, word_list: List[str], n_hidden: int = 1, hidden_size: int = 256):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """

        super().__init__()
        word_width = 26*5
        layers = [
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
        ]

        for _ in range(n_hidden):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, word_width))
        layers.append(nn.ReLU())

        self.f0 = nn.Sequential(*layers)

        word_array = np.zeros((word_width, len(word_list)))
        for i, word in enumerate(word_list):
            for j, c in enumerate(word):
                word_array[j*26 + (ord(c) - ord('A')), i] = 1

        self.words = torch.Tensor(word_array)
        self.actor_head = nn.Linear(word_width, word_width)

    def forward(self, x):

        y = self.f0(x.float())
        a = torch.clamp(torch.log_softmax(self.actor_head(y).view(-1, WORDLE_N, len(WORDLE_CHARS)), dim=-1), min=-10, max=0)

        # Shape of a is (batch_size, WORDLE_N, 26)

        return a

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index
