from typing import List

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch import Tensor
import re

class ActorCategorical(nn.Module):
    """Policy network, for discrete action spaces, which returns a distribution and an action given an
    observation."""

    def __init__(self, actor_net: nn.Module, word_list) -> None:
        """
        Args:
            actor_net: neural network that predicts action probabilities given the env state
        """
        super().__init__()

        self.actor_net = actor_net
        self.allowed_word_list = word_list

    def forward(self, states, pattern):
        logits = self.actor_net(states)

        if pattern is not None:
            temp_allowed_words_index = []
            new_word_list = []
            for i in range(len(self.allowed_word_list)):
                match = re.search(pattern, self.allowed_word_list[i])
                if match:
                    temp_allowed_words_index.append(i)
                    new_word_list.append(self.allowed_word_list[i])
            # import pdb;pdb.set_trace() 
            # self.allowed_word_list = new_word_list            # UNCOMMENT TO REDUCE THE ALLOWED WORD LIST. DO IT ONLY FOR FULL WORD LIST
            logits = logits[:,temp_allowed_words_index]

        pi = Categorical(logits=logits)
        actions = pi.sample()

        return pi, actions

    def get_log_prob(self, pi: Categorical, actions: Tensor):
        """Takes in a distribution and actions and returns log prob of actions under the distribution.

        Args:
            pi: torch distribution
            actions: actions taken by distribution

        Returns:
            log probability of the acition under pi
        """
        return pi.log_prob(actions)


class GreedyActorCategorical(nn.Module):
    
    def __init__(self, actor_net: nn.Module):
        super().__init__()

        self.actor_net = actor_net

    def forward(self, states: torch.Tensor, device: str) -> List[int]:
        """Takes in the current state and returns the action based on the agents policy.
        Args:
            states: current state of the environment
            device: the device used for the current batch
        Returns:
            action defined by policy
        """
        logits = self.actor_net(torch.tensor([states], device=device))
        probabilities = logits.exp().squeeze(dim=-1)
        prob_np = probabilities.data.cpu().numpy()

        actions = np.argmax(prob_np, axis=1)

        return list(actions)

    def get_log_prob(self, pi: Categorical, actions: Tensor):
        """Takes in a distribution and actions and returns log prob of actions under the distribution.

        Args:
            pi: torch distribution
            actions: actions taken by distribution

        Returns:
            log probability of the acition under pi
        """
        return pi.log_prob(actions)

