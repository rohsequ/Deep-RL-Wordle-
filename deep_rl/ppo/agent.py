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

    def __init__(self, actor_net: nn.Module) -> None:
        """
        Args:
            actor_net: neural network that predicts action probabilities given the env state
        """
        super().__init__()
        self.actor_net = actor_net

    def forward(self, states):
        # Shape of logits = (batch_size, WORDLE_N, 26)
        logits = self.actor_net(states)

        pi = Categorical(logits=logits)
        actions = pi.sample()

        # TODO: Check shape of actions
        # import pdb; pdb.set_trace()

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
        # import pdb; pdb.set_trace()

        actions = np.argmax(prob_np, axis=-1)

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
