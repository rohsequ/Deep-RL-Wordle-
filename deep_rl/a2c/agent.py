from typing import List

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import re
from wordle.const import WORDLE_N

class ActorCriticAgent:
    """Actor-Critic based agent that returns an action based on the networks policy."""

    def __init__(self, net, word_list):
        self.net = net
        self.allowed_word_list = word_list

    def __call__(self, states: torch.Tensor, device: str, pattern: str) -> List[int]:
        """Takes in the current state and returns the action based on the agents policy.
        Args:
            states: current state of the environment
            device: the device used for the current batch
            pattern: the word pattern needed to reduce the action space.
        Returns:
            action defined by policy
        """
        # Here, the logprobs are of (batch_size, 26 * WORDLE_N) shape. They give character_level logprobs
        logprobs, _ = self.net(torch.tensor(np.array([states]), device=device))
        probabilities = logprobs.exp().squeeze(dim=-1)
        prob_np = probabilities.data.cpu().numpy()

        # Here, prob_np should be of shape (batch_size, 26 * WORDLE_N)
        # Reshape it to be (batch_size, WORDLE_N, 26)
        prob_np = np.reshape(prob_np, (-1, WORDLE_N, 26))

        # TODO: Change this to select characters based on the probabilities
        # take the numpy values and randomly select characters based on prob distribution
        # Note that this is much faster than numpy.random.choice
        cdf = np.cumsum(prob_np, axis=-1)
        cdf[:, :, -1] = 1.  # Ensure cumsum adds to 1
        select = np.random.random((cdf.shape[0], cdf.shape[1]))
        actions = [
            np.searchsorted(cdf[row, col, :], select[row, col])
            for row in range(cdf.shape[0]) for col in range(cdf.shape[1])
        ]

        # TODO: Check the shape of actions. Would have to reshape it to give a five letter word

        return actions

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
        logits = self.actor_net(states)
        pi = Categorical(logits=logits)
        actions = pi.sample()

        return pi, actions


class GreedyActorCriticAgent:
    def __init__(self, net, word_list):
        self.net = net
        self.allowed_word_list = word_list

    def __call__(self, states: torch.Tensor, device: str, pattern: str) -> List[int]:
        """Takes in the current state and returns the action based on the agents policy.
        Args:
            states: current state of the environment
            device: the device used for the current batch
        Returns:
            action defined by policy
        """
        logprobs, _ = self.net(torch.tensor([states], device=device))
        probabilities = logprobs.exp().squeeze(dim=-1)
        prob_np = probabilities.data.cpu().numpy()
        
        # import pdb;pdb.set_trace()
        # TODO: Change this to character level rl
        actions = np.argmax(prob_np, axis=1)    
        
        return list(actions)