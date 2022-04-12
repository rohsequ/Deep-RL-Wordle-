import imp
from json.encoder import py_encode_basestring
from typing import List

import numpy as np
import torch
from torch import nn
from wordle.const import *
from wordle.const import WORDLE_N


class SumChars(nn.Module):
    def __init__(self, obs_size: int, word_list: List[str], n_hidden: int = 1, hidden_size: int = 256):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        word_width = 26*WORDLE_N
        layers = [
            nn.Linear(obs_size, hidden_size),
            nn.ReLU()
        ]
        for _ in range(n_hidden):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, word_width))
        layers.append(nn.ReLU())

        self.f0 = nn.Sequential(*layers)
        def init_normal(m):
            if type(m) == nn.Linear:
                nn.init.kaiming_normal_(m.weight)
        self.f0.apply(init_normal)
        
        word_array = np.zeros((word_width, len(word_list)))
        for i, word in enumerate(word_list):
            for j, c in enumerate(word):
                word_array[j*26 + (ord(c) - ord('A')), i] = 1
        self.words = torch.Tensor(word_array)

        self.actor_head = nn.Linear(word_width, word_width)
        self.critic_head = nn.Linear(word_width, 1)
        self.count = 0

    def forward(self, x):
        # import pdb;pdb.set_trace()
        eps = 0.001
        y = self.f0(x.float())
        y = (x - torch.mean(x)) / (torch.std(x) + eps)
        actor_a = torch.clamp(torch.log_softmax(
            torch.tensordot(self.actor_head(y),
                            self.words.to(self.get_device(y)),
                            dims=((1,), (0,))),
            dim=-1), min=-15, max=0)
        critic_c = self.critic_head(y)
        # if self.count >= 10000:
        # # # if -8 in actor_a:
            # import pdb; pdb.set_trace()
        self.count +=1
        
        return actor_a, critic_c

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index