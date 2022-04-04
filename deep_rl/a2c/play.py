from typing import Tuple, List

import wordle.state
from a2c.agent import GreedyActorCriticAgent
from a2c.module import AdvantageActorCritic
from wordle.wordle import WordleEnvBase


def load_from_checkpoint(
        checkpoint: str,
        evaluate: bool=True
) -> Tuple[AdvantageActorCritic, GreedyActorCriticAgent, WordleEnvBase]:
    """
    :param checkpoint:
    :return:
    """
    model = AdvantageActorCritic.load_from_checkpoint(checkpoint, evaluate=evaluate)
    env = model.env
    agent = GreedyActorCriticAgent(model.net, word_list=env.words)
    
    return model, agent, env


def suggest(
        agent: GreedyActorCriticAgent,
        env: WordleEnvBase,
        sequence: List[Tuple[str, List[int]]],
) -> str:
    """
    Given a list of words and masks, return the next suggested word

    :param agent:
    :param env:
    :param sequence: History of moves and outcomes until now
    :return:
    """
    state = env.reset()
    for word, mask in sequence:
        word = word.upper()
        assert word in env.words, f'{word} not in allowed words!'
        assert all(i in (0, 1, 2) for i in mask)
        assert len(mask) == 5

        state = wordle.state.update_from_mask(state, word, mask)

    return env.words[agent(state, "cpu", pattern=None)[0]]


def goal(
        agent: GreedyActorCriticAgent,
        env: WordleEnvBase,
        goal_word: str,
) -> Tuple[bool, List[Tuple[str, int]]]:
    state = env.reset()
    try:
        env.set_goal_word(goal_word.upper())
    except:
        raise ValueError("Goal word", goal_word, "not found in env words!")

    outcomes = []
    win = False
    for i in range(env.max_turns):
        dict_reduction_pattern = env.get_dict_reduce_pattern()
        action = agent(state, "cpu", pattern=dict_reduction_pattern)[0]
        state, reward, done, _ = env.step(action)
        outcomes.append((env.words[action], reward))
        if done:
            if reward >= 0:
                win = True
            break

    return win, outcomes
