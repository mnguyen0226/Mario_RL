"""
    Main.py function that called the module
    Test Run with 100 rounds
"""
import torch
from pathlib import Path

# Gym is an OpenAI toolkit for RL
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros

import datetime

###########################################################
from marioRL.Split.module.agent_dqn import * #Mario
from marioRL.Split.module.logger import MetricLogger
from marioRL.Split.module.preprocess import SkipFrame, GrayScaleObservation, ResizeObservation
###########################################################

def main():
    print("Running")
    # Initialize Super Mario environment => adept_main.py
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")

    # Limit the action-space to
    #   0. walk right
    #   1. jump right
    env = JoypadSpace(env, [["right"], ["right", "A"]])

    env.reset()
    next_state, reward, done, info = env.step(action=0)
    print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

    # Apply Wrappers to environment => adept_main.py
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)

    # TRAINING PROCESS => Run and call functions
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime(
        "%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n,
                  save_dir=save_dir)

    logger = MetricLogger(save_dir)

    episodes = 100  # Should be 40000
    for e in range(episodes):

        state = env.reset()

        # Play the game!
        while True:

            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            next_state, reward, done, info = env.step(action)

            # Remember
            mario.cache(state, next_state, action, reward, done)

            # Learn
            q, loss = mario.learn()

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            state = next_state

            # Check if end of game
            if done or info["flag_get"]:
                break

        logger.log_episode()

        if e % 20 == 0:  # Record anytime 20 episodes
            logger.record(episode=e, epsilon=mario.exploration_rate,
                          step=mario.curr_step)

        print(f"End {e} episode")
    print("Finish Training")


if __name__ == "__main__":
    main()