import numpy as np
import pandas as pd
import gym
import os
from typing import Optional, List
import h5py

dirname = os.path.dirname(__file__)
VALID_WORDS_PATH = f'{dirname}/../data/wordle_words.txt'


def _load_words(limit: Optional[int]=None) -> List[str]:
    with open(VALID_WORDS_PATH, 'r') as f:
        lines = [x.strip().upper() for x in f.readlines()]
        if not limit:
            return lines
        else:
            return lines[:limit]


def get_mask_from_state(state):
    state_str = ""
    dim = state.shape[0]
    
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    num_letters = 26
    word_len = word_len_var
    num_colors = 3

    used = []
    used_id = []

    for i in range(num_letters):

        if state[i+1]:
            used_id.append(i)
            used.append(letters[i])
    
    yes = []
    maybe = []
    no = []
    
    return used, yes, no, maybe



def checkdata(frames):

    states = frames["states"]
    actions = frames["actions"]
    dones = frames["dones"]
    returns = frames["returns"]
    targets = frames["targets"]

    batch_size = states.shape[0]
    print(batch_size)

    for i in range(800000, batch_size):
        s = states[i, :]
        a = actions[i]
        d = dones[i]
        t = targets[i]
        goal_word = word_list[t]
        action = word_list[a]
        s_data = get_mask_from_state(s)
        used = s_data[0]
        turns_left = s[0]

        print(turns_left, "goal: " + goal_word, "action: " + action, "used: ", used, "done? ", d)

        if d or turns_left == 1:
            print()
            input("Press Enter to go to next game...")

    # print(states.shape)
    # print(actions.shape)
    # print(dones[0])
    # print(returns[0])
    # print(targets[0])


keys = ["states", "actions", "dones", "returns", "targets"]
env_str = "WordleEnv1000-v0"

word_list = _load_words(1000)
file_name = "./data/" + env_str + ".hdf5"

with h5py.File(file_name, 'r') as f:
    checkdata(f)
