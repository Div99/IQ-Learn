from itertools import count
import torch
import gym
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from make_envs import make_env
from agent import make_agent
from utils.utils import eval_mode


def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(OmegaConf.to_yaml(cfg))
    return cfg


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)

    BATCH = args.train.batch
    EPISODE_STEPS = int(args.env.eps_steps)

    env = make_env(args)
    agent = make_agent(env, args)

    policy_file = 'experts'
    if args.eval.policy:
        policy_file = f'{args.eval.policy}'
    print(f'Loading policy from: {policy_file}', f'_{args.env.name}')

    agent.load(hydra.utils.to_absolute_path(policy_file), f'_{args.env.name}')

    episode_reward = 0
    evaluate(agent, env, num_episodes=args.eval.eps)

    if args.eval_only:
        exit()

    for epoch in count():
        state = env.reset()
        episode_reward = 0
        for episode_step in range(EPISODE_STEPS):
            env.render()
            action = agent.choose_action(state, sample=False)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            if done:
                break
            state = next_state
        print('Ep {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))


def evaluate(actor, env, num_episodes=10):
    """Evaluates the policy.
    Args:
      actor: A policy to evaluate.
      env: Environment to evaluate the policy on.
      num_episodes: A number of episodes to average the policy on.
    Returns:
      Averaged reward and a total number of steps.
    """
    total_timesteps = []
    total_returns = []

    for _ in range(num_episodes):
        eps_timesteps = 0
        eps_returns = 0

        state = env.reset()
        done = False
        while not done:
            with eval_mode(actor):
                action = actor.choose_action(state, sample=False)
            next_state, reward, done, info = env.step(action)

            if 'ale.lives' in info:  # true for breakout, false for pong
                done = info['ale.lives'] == 0

            eps_returns += reward
            eps_timesteps += 1
            state = next_state

        total_timesteps.append(eps_timesteps)
        total_returns.append(eps_returns)

    total_returns = np.array(total_returns)
    total_timesteps = np.array(total_timesteps)

    print("rewards: {:.2f} +/- {:.2f}".format(total_returns.mean(), total_returns.std()))
    print("len: {:.2f} +/- {:.2f}".format(total_timesteps.mean(), total_timesteps.std()))


if __name__ == "__main__":
    main()
