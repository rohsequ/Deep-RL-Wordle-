"""Advantage Actor Critic (A2C)"""
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb

from a2c.module import AdvantageActorCritic


def cli_main() -> None:
    parser = ArgumentParser(add_help=False)

    # trainer args
    parser = Trainer.add_argparse_args(parser)

    # model args
    parser = AdvantageActorCritic.add_model_specific_args(parser)
    args = parser.parse_args()

    with wandb.init(project='wordle-solver'):
        wandb.config.update(args)

        model = AdvantageActorCritic(**args.__dict__)

        # Wordle-100 200k iteration checkpoint
        # checkpoint = "/home/shreyas/EECS545/wordle-rl/wordle-solver/deep_rl/lightning_logs/version_16/checkpoints/epoch=0-step=199999.ckpt"

        # Wordle-1000 with warm start 500k iteration checkpoint
        # checkpoint = "/home/shreyas/EECS545/wordle-rl/wordle-solver/deep_rl/lightning_logs/version_21/checkpoints/epoch=0-step=449999.ckpt"
        # model_old = AdvantageActorCritic.load_from_checkpoint(checkpoint)


        checkpoint = "/home/shreyas/EECS545/current-code/Deep-RL-Wordle-/deep_rl/lightning_logs/version_13/checkpoints/epoch=0-step=199999.ckpt"
        model_old = AdvantageActorCritic.load_from_checkpoint(checkpoint)

        model.net.load_state_dict(model_old.net.state_dict())

        # save checkpoints based on avg_reward
        checkpoint_callback = ModelCheckpoint(every_n_train_steps=100)

        seed_everything(123)

        trainer = Trainer.from_argparse_args(args, callbacks=checkpoint_callback)
        # trainer = Trainer.from_argparse_args(args, deterministic=True, callbacks=checkpoint_callback)
        trainer.fit(model)


if __name__ == '__main__':
    cli_main()
