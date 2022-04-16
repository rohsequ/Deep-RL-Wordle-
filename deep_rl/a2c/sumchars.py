from typing import List
import torch
from torch import nn
from wordle.const import *
from wordle.const import WORDLE_N

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)

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
            # nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        ]
        for _ in range(n_hidden):
            layers.append(nn.Linear(hidden_size, hidden_size))
            # layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, word_width))
        # layers.append(nn.BatchNorm1d(word_width))
        layers.append(nn.ReLU())

        self.f0 = nn.Sequential(*layers)
        # self.f0.apply(init_normal)
        # word_array = np.zeros((word_width, len(word_list)))
        # for i, word in enumerate(word_list):
        #     for j, c in enumerate(word):
        #         word_array[j*26 + (ord(c) - ord('A')), i] = 1
        # self.words = torch.Tensor(word_array)

        self.actor_head = nn.Linear(word_width, word_width)
        # self.actor_head.apply(init_normal)
        self.critic_head = nn.Linear(word_width, 1)
        # self.critic_head.apply(init_normal)
        self.count = 0

    def forward(self, x):
        y = self.f0(x.float())

        # Currently outputting the logprobs of all the possible words in the dictionary
        # Instead, output character level logprobs
        # So, effectively, the 26 * WORDLE_N dimensional vector output of the actor_head
        actor_a = torch.clamp(torch.log_softmax(self.actor_head(y).view(-1, WORDLE_N, len(WORDLE_CHARS)), dim=-1), min=-10, max=0)
        critic_c = self.critic_head(y)

        self.count +=1
        
        return actor_a, critic_c

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index

class SumCharsLSTM(nn.Module):
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
        self.lstm = nn.LSTM(input_size=obs_size, hidden_size=hidden_size, num_layers=n_hidden, batch_first=True)

        layers = []
        layers.append(nn.Linear(hidden_size, word_width))
        # layers.append(nn.BatchNorm1d(word_width))
        layers.append(nn.ReLU())

        self.f0 = nn.Sequential(*layers)
        # self.f0.apply(init_normal)

        self.actor_head = nn.Linear(word_width, word_width)
        # self.actor_head.apply(init_normal)
        self.critic_head = nn.Linear(word_width, 1)
        # self.critic_head.apply(init_normal)
        self.count = 0

    def forward(self, x):
        inp = x.unsqueeze_(0)
        out_lstm, (self.h, self.c) = self.lstm(inp.float(), (self.h0, self.c0))
        y = self.f0(out_lstm)

        # Currently outputting the logprobs of all the possible words in the dictionary
        # Instead, output character level logprobs
        # So, effectively, the 26 * WORDLE_N dimensional vector output of the actor_head
        actor_a = torch.clamp(torch.log_softmax(self.actor_head(y).view(-1, WORDLE_N, len(WORDLE_CHARS)), dim=-1), min=-10, max=0)
        critic_c = self.critic_head(y)

        self.count +=1
        
        return actor_a, critic_c

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index

