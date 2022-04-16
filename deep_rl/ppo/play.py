from typing import Tuple, List

import wordle.state
from ppo.agent import GreedyActorCategorical
from ppo.module import PPO
from wordle.wordle import WordleEnvBase
from wordle.const import int2char

def load_from_checkpoint(
        checkpoint: str,
        evaluate: bool=True
) -> Tuple[PPO, GreedyActorCategorical, WordleEnvBase]:
    """
    :param checkpoint:
    :return:
    """
    model = PPO.load_from_checkpoint(checkpoint, evaluate=evaluate)
    agent = GreedyActorCategorical(model.net)
    env = model.env

    return model, agent, env

def goal(
        agent: GreedyActorCategorical,
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
        # dict_reduction_pattern = env.get_dict_reduce_pattern()
        action = agent(state, "cpu")[0]
        state, reward, done, _ = env.step(action)
        outcomes.append((''.join(int2char[a] for a in action), reward))
        if done:
            if reward >= 0:
                win = True
            break

    return win, outcomes
