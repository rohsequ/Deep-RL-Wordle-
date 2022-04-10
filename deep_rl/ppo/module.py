import collections
from argparse import ArgumentParser
from collections import OrderedDict
from typing import Any, List, Tuple, Iterator
import wandb

import gym
import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import Tensor, optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import ppo
import wordle.state
from ppo.agent import ActorCategorical
from ppo.experience import ExperienceSourceDataset, Experience

import h5py

from pl_bolts.utils import _GYM_AVAILABLE
from pl_bolts.models.rl.common.networks import MLP

class PPO(LightningModule):
    """PyTorch Lightning implementation of `Proximal Policy Optimization.
    <https://arxiv.org/abs/1707.06347>`_
    Paper authors: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov
    Model implemented by:
        `Sidhant Sundrani <https://github.com/sid-sundrani>`_
    Example:
        >>> from pl_bolts.models.rl.ppo_model import PPO
        >>> model = PPO("CartPole-v0")
    Note:
        This example is based on OpenAI's
        `PPO <https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py>`_ and
        `PPO2 <https://github.com/openai/baselines/blob/master/baselines/ppo2/ppo2.py>`_.
    Note:
        Currently only supports CPU and single GPU training with ``accelerator=dp``
    """

    def __init__(
        self,
        env: str,
        gamma: float = 0.99,
        lam: float = 0.95,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        max_episode_len: float = 200,
        batch_size: int = 512,
        steps_per_epoch: int = 2048,
        nb_optim_iters: int = 4,
        clip_ratio: float = 0.2,
        evaluate: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            env: gym environment tag
            gamma: discount factor
            lam: advantage discount factor (lambda in the paper)
            lr_actor: learning rate of actor network
            lr_critic: learning rate of critic network
            max_episode_len: maximum number interactions (actions) in an episode
            batch_size:  batch_size when training network- can simulate number of policy updates performed per epoch
            steps_per_epoch: how many action-state pairs to rollout for trajectory collection per epoch
            nb_optim_iters: how many steps of gradient descent to perform on each batch
            clip_ratio: hyperparameter for clipping in the policy objective
        """
        super().__init__()

        if not _GYM_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("This Module requires gym environment which is not installed yet.")

        # Hyperparameters
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.steps_per_epoch = steps_per_epoch
        self.nb_optim_iters = nb_optim_iters
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.max_episode_len = max_episode_len
        self.clip_ratio = clip_ratio
        self.save_hyperparameters()

        self.writer = SummaryWriter()

        self.env_str = env

        # Model components
        self.env = gym.make(env)
        self.net = ppo.construct(
            self.hparams.network_name,
            obs_size=self.env.observation_space.shape[0],
            n_hidden=self.hparams.n_hidden,
            hidden_size=self.hparams.hidden_size,
            word_list=self.env.words)

        # value network
        self.critic = MLP(self.env.observation_space.shape, 1)
        # policy network (agent)
        # actor_mlp = MLP(self.env.observation_space.shape, self.env.action_space.n)
        self.actor = ActorCategorical(self.net)

        self.batch_states = []
        self.batch_actions = []
        self.batch_adv = []
        self.batch_qvals = []
        self.batch_logp = []
        self.batch_masks = []
        self.batch_targets = []

        self.ep_rewards = []
        self.ep_values = []
        self.epoch_rewards = []

        # Tracking metrics
        self.episode_reward = 0
        self.done_episodes = 0
        self.eps = np.finfo(np.float32).eps.item()

        self._winning_steps = 0
        self._winning_rewards = 0
        self._total_rewards = 0
        self._wins = 0
        self._losses = 0
        self._last_win = []
        self._last_loss = []
        self._seq = []

        self._recent_losing_words = collections.deque(maxlen=1000)
        self._cheat_word = None

        self.episode_step = 0
        self.avg_ep_reward = 0
        self.avg_ep_len = 0
        self.avg_reward = 0

        self.state = self.env.reset()

        # For collecting data
        self._num_batches_before_clear = 10
        self._resize_dset = False
        self._data = {"states": [], "actions": [], "dones": [], "qvals": [], "adv": [], "targets" : []}

        # Create the hd5 file
        if not evaluate:
            file_name = "./data/ppo/" + self.env_str + ".hdf5"
            sz = self._num_batches_before_clear * self.steps_per_epoch
            with h5py.File(file_name, 'w') as f:
                states_dset = f.create_dataset("states", (sz, self.state.shape[0]), maxshape=(None, 417),dtype=np.uint, compression="gzip", compression_opts=9)
                actions_dset = f.create_dataset("actions", (sz,), maxshape=(None,),dtype=np.uint, compression="gzip", compression_opts=9)
                dones_dset = f.create_dataset("dones", (sz,), maxshape=(None,),dtype=np.bool_, compression="gzip", compression_opts=9)
                qvals_dset = f.create_dataset("qvals", (sz,), maxshape=(None,),dtype=np.float, compression="gzip", compression_opts=9)
                adv_dset = f.create_dataset("adv", (sz,), maxshape=(None,),dtype=np.float, compression="gzip", compression_opts=9)
                targets_dset = f.create_dataset("targets", (sz,), maxshape=(None,),dtype=np.uint, compression="gzip", compression_opts=9)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Passes in a state x through the network and returns the policy and a sampled action.
        Args:
            x: environment state
        Returns:
            Tuple of policy and action
        """
        
        pi, action = self.actor(torch.FloatTensor([x], device=self.device))
        value = self.critic(torch.FloatTensor([x], device=self.device))

        return pi, action, value

    def discount_rewards(self, rewards: List[float], discount: float) -> List[float]:
        """Calculate the discounted rewards of all rewards in list.
        Args:
            rewards: list of rewards/advantages
            discount: discount factor
        Returns:
            list of discounted rewards/advantages
        """
        assert isinstance(rewards[0], float)

        cumul_reward = []
        sum_r = 0.0

        for r in reversed(rewards):
            sum_r = (sum_r * discount) + r
            cumul_reward.append(sum_r)

        return list(reversed(cumul_reward))

    def calc_advantage(self, rewards: List[float], values: List[float], last_value: float) -> List[float]:
        """Calculate the advantage given rewards, state values, and the last value of episode.
        Args:
            rewards: list of episode rewards
            values: list of state values from critic
            last_value: value of last state of episode
        Returns:
            list of advantages
        """
        rews = rewards + [last_value]
        vals = values + [last_value]
        # GAE
        delta = [rews[i] + self.gamma * vals[i + 1] - vals[i] for i in range(len(rews) - 1)]
        adv = self.discount_rewards(delta, self.gamma * self.lam)

        return adv

    def generate_trajectory_samples(self) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Contains the logic for generating trajectory data to train policy and value network.
        Yield:
           Tuple of Lists containing tensors for states, actions, log probs, qvals and advantage
        """

        for step in range(self.steps_per_epoch):

            with torch.no_grad():
                pi, action, value = self(self.state)
                log_prob = self.actor.get_log_prob(pi, action[0])
            
            if wordle.state.remaining_steps(self.state) == 1 and self._cheat_word:
                action = torch.tensor([self._cheat_word])
            
            next_state, reward, done, aux = self.env.step(action[0])
            reward = float(reward)

            self.episode_step += 1

            self.batch_states.append(self.state)
            self.batch_actions.append(action[0])
            self.batch_logp.append(log_prob)
            self.batch_masks.append(done)
            self.batch_targets.append(aux['goal_id'])

            self._seq.append(Experience(self.state.copy(), action[0], reward, aux['goal_id']))

            self.ep_rewards.append(reward)
            self.ep_values.append(value.item())

            self.state = next_state
            
            for _ in range(self.hparams.batch_size):
                dict_reduction_pattern = self.env.get_dict_reduce_pattern()
                action = self.agent(self.state, self.device, dict_reduction_pattern)[0]
                if wordle.state.remaining_steps(self.state) == 1 and self._cheat_word:
                    action = self._cheat_word

            epoch_end = step == (self.steps_per_epoch - 1)
            terminal = len(self.ep_rewards) == self.max_episode_len

            if epoch_end or done or terminal:
                # if trajectory ends abtruptly, boostrap value of next state
                if (terminal or epoch_end) and not done:

                    with torch.no_grad():
                        _, _, value = self(self.state)
                        last_value = value.item()
                        steps_before_cutoff = self.episode_step
                else:
                    last_value = 0
                    steps_before_cutoff = 0

                # discounted cumulative reward
                self.batch_qvals += self.discount_rewards(self.ep_rewards + [last_value], self.gamma)[:-1]
                # advantage
                self.batch_adv += self.calc_advantage(self.ep_rewards, self.ep_values, last_value)
                # logs
                self.epoch_rewards.append(sum(self.ep_rewards))
                # reset params
                self.ep_rewards = []
                self.ep_values = []
                self.episode_step = 0
                self.state = self.env.reset()

                if done:
                    if action == self.env.goal_word:
                        self._winning_steps += self.env.max_turns - wordle.state.remaining_steps(self.state)
                        self._wins += 1
                        self._winning_rewards += self.epoch_rewards[-1]
                        self._last_win = self._seq
                    else:
                        self._losses += 1
                        self._last_loss = self._seq
                        self._recent_losing_words.append(aux['goal_id'])

                    self._seq = []
                    self._total_rewards += self.episode_reward

                    self.done_episodes += 1
                    # With some probability, override the word with one that we lost recently
                    self._cheat_word = None
                    if len(self._recent_losing_words) > 0:
                        if np.random.random() < self.hparams.prob_play_lost_word:
                            lost_idx = int(np.random.random()*len(self._recent_losing_words))
                            self.env.set_goal_id(self._recent_losing_words[lost_idx])
                            if np.random.random() < self.hparams.prob_cheat:
                                self._cheat_word = self._recent_losing_words[lost_idx]

                    self.episode_reward = 0

            if epoch_end:

                self._data["states"].extend(self.batch_states)
                self._data["actions"].extend([action.item() for action  in self.batch_actions])
                self._data["dones"].extend(self.batch_masks)
                self._data["qvals"].extend(self.batch_qvals)
                self._data["adv"].extend(self.batch_adv)
                self._data["targets"].extend(self.batch_targets)

                if len(self._data["actions"]) >= self._num_batches_before_clear * len(self.batch_actions):
                    
                    length = len(self._data["actions"])

                    file_name = "./data/ppo/" + self.env_str + ".hdf5"
                    with h5py.File(file_name, 'a') as f:

                        states_dset = f["states"]
                        actions_dset = f["actions"]
                        dones_dset = f["dones"]
                        qvals_dset = f["qvals"]
                        adv_dset = f["adv"]
                        targets_dset = f["targets"]

                        curr_size = states_dset.shape[0]

                        if self._resize_dset:
                            states_dset.resize(curr_size + length, axis=0)
                            actions_dset.resize(curr_size + length, axis=0)
                            dones_dset.resize(curr_size + length, axis=0)
                            qvals_dset.resize(curr_size + length, axis=0)
                            adv_dset.resize(curr_size + length, axis=0)
                            targets_dset.resize(curr_size + length, axis=0)

                            states_dset[curr_size:, :] = self._data["states"]
                            actions_dset[curr_size:] = self._data["actions"]
                            dones_dset[curr_size:] = self._data["dones"]
                            qvals_dset[curr_size:] = self._data["qvals"]
                            adv_dset[curr_size:] = self._data["adv"]
                            targets_dset[curr_size:] = self._data["targets"]

                        else:
                            self._resize_dset = True
                            states_dset[:, :] = self._data["states"]
                            actions_dset[:] = self._data["actions"]
                            dones_dset[:] = self._data["dones"]
                            qvals_dset[:] = self._data["qvals"]
                            adv_dset[:] = self._data["adv"]
                            targets_dset[:] = self._data["targets"]

                    # Free up memory
                    for k in self._data:
                        self._data[k] = []

                train_data = zip(
                    self.batch_states, self.batch_actions, self.batch_logp, self.batch_qvals, self.batch_adv
                )

                for state, action, logp_old, qval, adv in train_data:
                    yield state, action, logp_old, qval, adv

                self.batch_states.clear()
                self.batch_actions.clear()
                self.batch_adv.clear()
                self.batch_logp.clear()
                self.batch_qvals.clear()
                self.batch_masks.clear()
                self.batch_targets.clear()
                
                

                # logging
                self.avg_reward = sum(self.epoch_rewards) / self.steps_per_epoch

                # if epoch ended abruptly, exlude last cut-short episode to prevent stats skewness
                epoch_rewards = self.epoch_rewards
                if not done:
                    epoch_rewards = epoch_rewards[:-1]

                total_epoch_reward = sum(epoch_rewards)
                nb_episodes = len(epoch_rewards)

                self.avg_ep_reward = total_epoch_reward / nb_episodes
                self.avg_ep_len = (self.steps_per_epoch - steps_before_cutoff) / nb_episodes

                self.epoch_rewards.clear()

    def actor_loss(self, state, action, logp_old, adv) -> Tensor:
        pi, _ = self.actor(state)
        logp = self.actor.get_log_prob(pi, action)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_actor = -(torch.min(ratio * adv, clip_adv)).mean()
        return loss_actor

    def critic_loss(self, state, qval) -> Tensor:
        value = self.critic(state)
        loss_critic = (qval - value).pow(2).mean()
        return loss_critic

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx, optimizer_idx):
        """Carries out a single update to actor and critic network from a batch of replay buffer.
        Args:
            batch: batch of replay buffer/trajectory data
            batch_idx: not used
            optimizer_idx: idx that controls optimizing actor or critic network
        Returns:
            loss
        """
        state, action, old_logp, qval, adv = batch

        # normalize advantages
        adv = (adv - adv.mean()) / adv.std()

        self.log("avg_ep_len", self.avg_ep_len, prog_bar=True, on_step=False, on_epoch=True)
        self.log("avg_ep_reward", self.avg_ep_reward, prog_bar=True, on_step=False, on_epoch=True)
        self.log("avg_reward", self.avg_reward, prog_bar=True, on_step=False, on_epoch=True)

        if optimizer_idx == 0:
            loss_actor = self.actor_loss(state, action, old_logp, adv)
            self.log("loss_actor", loss_actor, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            return loss_actor

        if optimizer_idx == 1:
            loss_critic = self.critic_loss(state, qval)
            self.log("loss_critic", loss_critic, on_step=False, on_epoch=True, prog_bar=False, logger=True)

            return loss_critic

        if self.current_epoch % 50 == 0:
            metrics = {
                "train_loss_actor": loss_actor,
                "train_loss_critic": loss_critic,
                "total_games_played": self.done_episodes,
                "lose_ratio": self._losses/(self._wins+self._losses),
                "wins": self._wins,
                "reward_per_game": self._total_rewards / (self._wins+self._losses),
                "global_step": self.global_step,
            }

            if self._wins > 0:
                metrics["reward_per_win"] = self._winning_rewards / self._wins
                metrics["avg_winning_turns"] = self._winning_steps / self._wins

            for k, v in metrics.items():
                self.writer.add_scalar(k, v, global_step=self.global_step)

            def get_game_string(seq):
                game = f'goal: {self.env.words[seq[0].goal_id]}\n'
                for i, exp in enumerate(seq):
                    game += f'{i}: {self.env.words[exp.action]}\n'
                return game

            def get_table_row(seq):
                goal = self.env.words[seq[0].goal_id]
                guesses = ""
                for i, exp in enumerate(seq):
                    guesses += f'{i}: {self.env.words[exp.action]} '
                return [goal, guesses]

            if len(self._last_win):
                self.writer.add_text("last_win", get_game_string(self._last_win), global_step=self.global_step)
                metrics["last_win"] = wandb.Table(data=[get_table_row(self._last_win)], columns=['goal', 'guesses'])
            if len(self._last_loss):
                self.writer.add_text("last_loss", get_game_string(self._last_loss), global_step=self.global_step)
                metrics["last_loss"] = wandb.Table(data=[get_table_row(self._last_loss)], columns=['goal', 'guesses'])

            wandb.log(metrics)


        raise NotImplementedError(
            f"Got optimizer_idx: {optimizer_idx}. Expected only 2 optimizers from configure_optimizers. "
            "Modify optimizer logic in training_step to account for this. "
        )

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        return optimizer_actor, optimizer_critic

    def optimizer_step(self, *args, **kwargs):
        """Run ``nb_optim_iters`` number of iterations of gradient descent on actor and critic for each data
        sample."""
        for _ in range(self.nb_optim_iters):
            super().optimizer_step(*args, **kwargs)

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = ExperienceSourceDataset(self.generate_trajectory_samples)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self._dataloader()

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument("--env", type=str, default="WordleEnv100-v0")
        parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
        parser.add_argument("--lam", type=float, default=0.95, help="advantage discount factor")
        parser.add_argument("--lr_actor", type=float, default=3e-4, help="learning rate of actor network")
        parser.add_argument("--lr_critic", type=float, default=1e-3, help="learning rate of critic network")
        parser.add_argument("--max_episode_len", type=int, default=1000, help="capacity of the replay buffer")
        parser.add_argument("--batch_size", type=int, default=512, help="batch_size when training network")
        parser.add_argument(
            "--steps_per_epoch",
            type=int,
            default=2048,
            help="how many action-state pairs to rollout for trajectory collection per epoch",
        )
        parser.add_argument(
            "--nb_optim_iters", type=int, default=4, help="how many steps of gradient descent to perform on each batch"
        )
        parser.add_argument(
            "--clip_ratio", type=float, default=0.2, help="hyperparameter for clipping in the policy objective"
        )
        parser.add_argument("--network_name", type=str, default="SumChars", help="Network to use")
        parser.add_argument("--n_hidden", type=int, default="1", help="Number of hidden layers")
        parser.add_argument("--hidden_size", type=int, default="256", help="Width of hidden layers")
        parser.add_argument("--seed", type=int, default=123, help="seed for training run")
        parser.add_argument("--prob_play_lost_word", type=float, default=0, help="Probabiilty of replaying a losing word")
        parser.add_argument("--prob_cheat", type=float, default=0, help="Probability of cheating when playing lost word")
        parser.add_argument("--weight_decay", type=float, default=0., help="Optimizer weight decay regularization.")

        parser.add_argument(
            "--avg_reward_len",
            type=int,
            default=100,
            help="how many episodes to include in avg reward",
        )

        parser.add_argument("--evaluate", type=bool, default=False, help="Whether the model is in evaluate mode.")
        
        return parser
