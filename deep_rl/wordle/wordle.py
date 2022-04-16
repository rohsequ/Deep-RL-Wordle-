# from collections import OrderedDict
import os
from typing import Optional, List 

import gym
from gym import spaces
import numpy as np
import pandas as pd
import wordle.state
import random
from wordle.const import *

CUR_PATH = os.environ.get('PYTHONPATH', '.')
import os
dirname = os.path.dirname(__file__)
# VALID_WORDS_PATH = f'{dirname}/../../data/words.csv'
VALID_WORDS_PATH = f'{dirname}/../../data/wordle_words.txt'


# def _load_words(limit: Optional[int]=None) -> List[str]:
#     w_bank = pd.read_csv(VALID_WORDS_PATH)
#     w_bank = w_bank[w_bank['words'].str.len() == WORDLE_N]
#     lines = w_bank['words'].str.upper().tolist()
#     # random.shuffle(lines)

#     if not limit:
#         return lines
#     else:
#         return lines[:limit]

def _load_words(limit: Optional[int]=None) -> List[str]:
    with open(VALID_WORDS_PATH, 'r') as f:
        lines = [x.strip().upper() for x in f.readlines()]
        if not limit:
            return lines
        else:
            return lines[:limit]


class WordleEnvBase(gym.Env):
    """
    Actions:
        Can play any 5 letter word in vocabulary
        * 13k for full vocab
    State space is defined as:
    TO BE DEFINED - Hardik
    """
    def __init__(self, words: List[str],
                 word_n: int,
                 max_turns: int,
                 allowable_words: Optional[int] = None):

        global WORDLE_N
        WORDLE_N = word_n

        assert all(len(w) == WORDLE_N for w in words), f'Not all words of length {WORDLE_N}, {words}'
        self.words = words
        self.max_turns = max_turns
        self.allowable_words = allowable_words
        # self.mask_based_state_updates = mask_based_state_updates
        if not self.allowable_words:
            self.allowable_words = len(self.words)

        # self.action_space = spaces.Discrete(len(self.words))

        self.action_space = spaces.MultiDiscrete([len(WORDLE_CHARS)] * WORDLE_N)
        self.observation_space = spaces.MultiDiscrete(wordle.state.get_nvec(self.max_turns))

        # Rewards Calculations 
        #####################################
        self.reward = 0
        self.color = np.zeros(WORDLE_N)
        self.color_vec = np.zeros(len(WORDLE_CHARS))
        self.pattern = None
        #####################################

        self.done = True
        self.goal_word: int = -1
        self.state: wordle.state.WordleState = None
        self.state_updater = wordle.state.update

        self.int2char = {k: c for k, c in enumerate(WORDLE_CHARS)}

    def step(self, action: list[int]):
        """
        Here action should be a list of length WORDLE_N, giving the integer of the character at each position 
        """
        
        if self.done:
            raise ValueError(
                "You are calling 'step()' even though this "
                "environment has already returned done = True. You "
                "should always call 'reset()' once you receive 'done = "
                "True' -- any further steps are undefined behavior."
            )
        
        action_word = ''.join(self.int2char[a] for a in action)

        # Before updating, check if we are using letters that are already black
        # Add big negative rewards for that
        for cint in action:
            if self.state[1 + cint]: # Character has been used before
                offset = 1 + len(WORDLE_CHARS) + cint
                for i in range(WORDLE_N):
                    if self.state[offset + len(WORDLE_CHARS) * i] == GREY:
                        self.reward += -3
                        break

        self.state, self.color = self.state_updater(state=self.state,
                                        word=action_word,
                                        goal_word=self.words[self.goal_word])

        # self.set_dict_reduce_pattern(self.words[action])

        for cint, col in zip(action, self.color):
            # cint = ord(char) - ord(WORDLE_CHARS[0])
            if self.color_vec[cint] == 0:
                if col == GREEN:
                    self.reward += GREEN_REWARD
                elif col == YELLOW:
                    self.reward += YELLOW_REWARD
                elif col == GREY:
                    self.reward += GREY_REWARD
                self.color_vec[cint] = col

            elif self.color_vec[cint] == YELLOW:
                if col == GREEN:
                    self.reward += GREEN_REWARD
                    self.color_vec[cint] = col

        if action_word == self.words[self.goal_word]:
            self.done = True
            #reward = REWARD
            if wordle.state.remaining_steps(self.state) == self.max_turns-1:
                self.reward = 0   # No reward for guessing off the bat
            else:
                # reward = REWARD*(self.state.remaining_steps() + 1) / self.max_turns
                self.reward += WIN_REWARD
                # self.reward = 10
        
        # elif wordle.state.remaining_steps(self.state) == 0:
        #     self.done = True
        #     self.reward = -10 # Cumulative loss. Includes the step loss for max turns.

        elif wordle.state.remaining_steps(self.state) == 0:
            self.done = True
            self.reward += LOSS # Cumulative loss. Includes the step loss for max turns.
            
        else:
            if wordle.state.remaining_steps(self.state) < self.max_turns-2:
                # self.reward += (self.max_turns - wordle.state.remaining_steps(self.state))*STEP_LOSS
                self.reward += STEP_LOSS

        return self.state.copy(), self.reward, self.done, {"goal_id": self.goal_word}

    def reset(self, seed: Optional[int] = None):
        self.state = wordle.state.new(self.max_turns)
        self.done = False
        self.goal_word = int(np.random.random()*self.allowable_words)

        self.reward = 0
        self.color = np.zeros(WORDLE_N)
        self.color_vec = np.zeros(len(WORDLE_CHARS))
        self.pattern = None

        return self.state.copy()

    def set_goal_word(self, goal_word: str):
        self.goal_word = self.words.index(goal_word)

    def set_goal_id(self, goal_id: int):
        self.goal_word = goal_id
    
    # def get_dict_reduce_pattern(self):
    #     return self.pattern
    
    # def set_dict_reduce_pattern(self, action):
    #     color_invert = OrderedDict()
    #     unique_letters = {''}
    #     for k, v in (zip(action, self.color)):
    #         if v not in color_invert.keys():
    #             unique_letters.add(k)
    #             if v == YELLOW:
    #                 color_invert[v] = f"(?=.*{k}.*)"
    #             if v == GREY:
    #                 color_invert[v] = f"[^{k}<next>]"
    #         else:
    #             if k not in unique_letters:
    #                 unique_letters.add(k)
    #                 if v == YELLOW:
    #                     color_invert[v]+=(f"(?=.*{k}.*)")
    #                 if v == GREY:
    #                     color_invert[v] = color_invert[v].replace("<next>", f"{k}<next>")

    #     if YELLOW not in color_invert.keys():
    #         color_invert[YELLOW] = ""
    #     if GREY not in color_invert.keys():
    #         color_invert[GREY] = "[A-Z]"
    #     else:
    #         color_invert[GREY] = color_invert[GREY].replace("<next>","")

    #     self.pattern = "<green_grey_letters><yellow_letters>"
    #     word_pattern = "(?=<word>)"


    #     for key, val in (zip(action, self.color)):
    #         if val == GREEN:
    #             word_pattern = word_pattern.replace("<word>", f"[{key}]<word>")
    #         if val == YELLOW:
    #             word_pattern = word_pattern.replace("<word>", f"{color_invert[GREY]}<word>")
    #         if val == GREY:
    #             word_pattern = word_pattern.replace("<word>", f"{color_invert[GREY]}<word>")

    #     word_pattern = word_pattern.replace("<word>", "")
    #     self.pattern = self.pattern.replace("<green_grey_letters>", word_pattern)
    #     self.pattern = self.pattern.replace("<yellow_letters>", f"{color_invert[YELLOW]}")
    #     return self.pattern


class WordleEnv10_4(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(10), max_turns=5, word_n=4)


class WordleEnv10(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(10), max_turns=6, word_n=4)


class WordleEnv100(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(100), max_turns=6, word_n=4)


class WordleEnv100OneAction(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(100), allowable_words=1, max_turns=6, word_n=4)


class WordleEnv100WithMask(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(100), max_turns=6,
                         mask_based_state_updates=True)


class WordleEnv100TwoAction(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(100), allowable_words=2, max_turns=6, word_n=4)


class WordleEnv100FullAction(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(), allowable_words=100, max_turns=6)


class WordleEnv1000(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(1000), max_turns=6, word_n=4)


class WordleEnv1000WithMask(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(1000), max_turns=6,
                         mask_based_state_updates=True)


class WordleEnv1000FullAction(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(), allowable_words=1000, max_turns=6)


class WordleEnvFull(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(), max_turns=6, word_n=4)


class WordleEnvReal(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(), allowable_words=2315, max_turns=6, word_n=4)


class WordleEnvRealWithMask(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(), allowable_words=2315, max_turns=6,
                         mask_based_state_updates=True)

class WordleEnv5000(WordleEnvBase):
    def __init__(self):
        super().__init__(words=_load_words(5000), max_turns=6, word_n=4)
