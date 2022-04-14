import imp
from json.encoder import py_encode_basestring
from typing import List
from unicodedata import bidirectional

import numpy as np
import torch
from torch import nn
from wordle.const import *


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
        self.h0 = torch.randn(n_hidden, 1, hidden_size) # D*numlayers, Hout D = 1 for uni-directional
        self.c0 = torch.randn(n_hidden, 1, hidden_size) # D*numlayers, Hout
        self.f1 = nn.LSTM(input_size=obs_size, hidden_size=hidden_size, num_layers=n_hidden, batch_first=True)
        self.f2 = nn.Linear(hidden_size, word_width)
        nn.init.kaiming_normal_(self.f2.weight)
        self.f3 = nn.ReLU()
        
        word_array = np.zeros((word_width, len(word_list)))
        for i, word in enumerate(word_list):
            for j, c in enumerate(word):
                word_array[j*26 + (ord(c) - ord('A')), i] = 1
        self.words = torch.Tensor(word_array)

        self.actor_head = nn.Linear(word_width, word_width)
        self.critic_head = nn.Linear(word_width, 1)
        self.count = 0

    def forward(self, x):
        # import pdb; pdb.set_trace()
        input = x.unsqueeze_(0) # N, L, Hin
        out_lstm, (self.h0, self.c0) = self.f1(input.float(), (self.h0, self.c0))
        out_linear = self.f2(out_lstm.squeeze(0))
        y = self.f3(out_linear)
        actor_a = torch.clamp(torch.log_softmax(
            torch.tensordot(self.actor_head(y),
                            self.words.to(self.get_device(y)),
                            dims=((1,), (0,))),
            dim=-1), min=-15, max=0)
        critic_c = self.critic_head(y)
        self.count +=1
        
        return actor_a, critic_c

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index