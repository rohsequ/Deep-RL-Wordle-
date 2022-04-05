from collections import OrderedDict
import numpy as np

from wordle.const import WORDLE_CHARS, WORDLE_N, GREEN, YELLOW, GREY, C_UNK


WordleState = np.ndarray

def set_wordle_word_len(word_len_var: int):
    return WORDLE_N = word_len_var

def get_nvec(max_turns: int):
    return [max_turns] + [2] * len(WORDLE_CHARS) + [4] * WORDLE_N * len(WORDLE_CHARS)


def new(max_turns: int) -> WordleState:
    return np.array(
        [max_turns] + [0] * len(WORDLE_CHARS) + [C_UNK] * WORDLE_N * len(WORDLE_CHARS),
        dtype=np.float64)


def remaining_steps(state: WordleState) -> int:
    return state[0]


def update(state: WordleState, word: str, goal_word: str) -> WordleState:
    state = state.copy()
    state[0] -= 1
    tmp_goal_word = goal_word
    color = np.zeros(WORDLE_N)
    for i, c in enumerate(word):
        cint = ord(c) - ord(WORDLE_CHARS[0])
        offset = 1 + len(WORDLE_CHARS) + cint
        state[1 + cint] = 1
        if goal_word[i] == c:
            color[i] = GREEN
            # char at position i = yes, all other chars at position i == no
            state[offset + len(WORDLE_CHARS)*i] = GREEN
            for ocint in range(len(WORDLE_CHARS)):
                if ocint != cint:
                    oc_offset = 1 + len(WORDLE_CHARS) + ocint 
                    state[oc_offset + len(WORDLE_CHARS)*i] = GREY
            for nwcint in range(len(word)):
                if (nwcint != i) and (state[offset + len(WORDLE_CHARS)*nwcint] != GREEN) and (state[offset + len(WORDLE_CHARS)*nwcint] != GREY):   
                    state[offset + len(WORDLE_CHARS)*nwcint] = YELLOW
                    
            tmp_goal_word = tmp_goal_word.replace(c, '', 1) 
        elif c in tmp_goal_word:
            color[i] = YELLOW
            # Char at position i = no, other chars stay as they are
            state[offset + len(WORDLE_CHARS)*i] = GREY
            for nwcint in range(len(word)):
                if (nwcint != i) and (state[offset + len(WORDLE_CHARS)*nwcint] != GREEN) and (state[offset + len(WORDLE_CHARS)*nwcint] != GREY):  
                    state[offset + len(WORDLE_CHARS)*nwcint] = YELLOW
        else:
            color[i] = GREY
            # Char at all positions = no
            for nwcint in range(len(word)):
                state[offset + len(WORDLE_CHARS)*nwcint] = GREY
    return state, color

