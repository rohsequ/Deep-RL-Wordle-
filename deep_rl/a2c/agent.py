from typing import List

import numpy as np
import torch
from wordle.const import WORDLE_N

class ActorCriticAgent:
    """Actor-Critic based agent that returns an action based on the networks policy."""

    def __init__(self, net, word_list):
        self.net = net
        self.allowed_word_list = word_list

    def __call__(self, states: torch.Tensor, device: str) -> List[int]:
        """Takes in the current state and returns the action based on the agents policy.
        Args:
            states: current state of the environment
            device: the device used for the current batch
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
            [np.searchsorted(cdf[row, col, :], select[row, col])
            for col in range(cdf.shape[1])] for row in range(cdf.shape[0])
        ]

        # Actions is a list with a list as its only element. The items of this list are the 5 ints corresponding to 5 letters
        # import pdb; pdb.set_trace()

        return actions

class GreedyActorCriticAgent:
    def __init__(self, net, word_list):
        self.net = net
        self.allowed_word_list = word_list

    def __call__(self, states: torch.Tensor, device: str) -> List[int]:
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
        
        # Here, prob_np should be of shape (batch_size, 26 * WORDLE_N)
        # Reshape it to be (batch_size, WORDLE_N, 26)
        prob_np = np.reshape(prob_np, (-1, WORDLE_N, 26))

        actions = np.argmax(prob_np, axis=-1)

        return list(actions)