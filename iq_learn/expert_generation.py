import random
from itertools import count
from collections import defaultdict

import gym
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter

from agent import make_agent
from make_envs import make_env, is_atari
from dataset.memory import Memory
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(OmegaConf.to_yaml(cfg))
    return cfg


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)

    env = make_env(args)
    if args.eval.use_baselines:
        from baselines_zoo.baselines_expert import BaselinesExpert
        agent = BaselinesExpert(args.env.name, folder='rl-trained-agents')
        # env = agent.env
    else:
        agent = make_agent(env, args)

    expert_file = f'{args.method.type}.para'
    if args.eval.policy:
        expert_file = f'{args.eval.policy}'
    print(f'Loading expert from: {expert_file}')

    agent.load(hydra.utils.to_absolute_path(expert_file), f'_{args.env.name}')

    episode_reward = 0

    REPLAY_MEMORY = None
    REWARD_THRESHOLD = args.eval.threshold
    MAX_EPS = args.expert.demos
    EPS_STEPS = int(args.env.eps_steps)

    memory_replay = Memory(REPLAY_MEMORY)
    total_steps = 0
    saved_eps = 0
    expert_trajs = defaultdict(list)
    expert_lengths = []
    expert_rewards = []

    for epoch in count():
        if saved_eps >= MAX_EPS:
            break

        state = env.reset()
        episode_reward = 0
        traj = []

        episode_infos = None
        for time_steps in range(EPS_STEPS):
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            if is_atari(args.env.name) and isinstance(action, np.ndarray):
                action = action.item()

            traj.append((state, next_state, action, reward, done))

            episode_reward += reward
            if memory_replay.size() == REPLAY_MEMORY:
                print('expert replay saved...')
                memory_replay.save(f'experts/{args.env_name}_{args.expert.demos}')
                exit()

            state = next_state

            if is_atari(args.env.name):
                if 'ale.lives' in info:  # true for breakout, false for pong
                    done = info['ale.lives'] == 0
                episode_infos = info.get("episode")
                if episode_infos is not None:
                    episode_reward = episode_infos['r']
                    print("Atari Episode Length", episode_infos["l"])

            if done:
                use_success = 'is_success' in info.keys()
                score = info.get('is_success')
                break

        if (not REWARD_THRESHOLD or episode_reward >= REWARD_THRESHOLD) and (not use_success or score >= 1.):
            saved_eps += 1
            states, next_states, actions, rewards, dones = zip(*traj)

            expert_trajs["states"].append(states)
            expert_trajs["next_states"].append(next_states)
            expert_trajs["actions"].append(actions)
            expert_trajs["rewards"].append(rewards)
            expert_trajs["dones"].append(dones)
            expert_trajs["lengths"].append(len(traj))
            expert_lengths.append(len(traj))
            expert_rewards.append(episode_reward)
            print('Ep {}\tSaving Episode reward: {:.2f}\t'.format(epoch, episode_reward))
        else:
            print('Ep {}\tSkipped episode with reward: {:.2f}\t'.format(epoch, episode_reward))

    # for k, v in expert_trajs.items():
    #     expert_trajs[k] = np.array(v)

    get_data_stats(expert_trajs, np.array(expert_rewards), np.array(expert_lengths))

    print('Final size of Replay Buffer: {}'.format(sum(expert_trajs["lengths"])))
    with open(hydra.utils.to_absolute_path(f'experts/{args.env.name}_{args.expert.demos}.pkl'), 'wb') as f:
        pickle.dump(expert_trajs, f)
    exit()


def get_data_stats(d, rewards, lengths):
    # lengths = d["lengths"]

    print("rewards: {:.2f} +/- {:.2f}".format(rewards.mean(), rewards.std()))
    print("len: {:.2f} +/- {:.2f}".format(lengths.mean(), lengths.std()))


def padded(a, target_length, axis=0):
    """Add padding at end of to make array fixed size"""
    x = np.array(a)
    pad_size = target_length - x.shape[axis]
    axis_nb = len(x.shape)

    if pad_size < 0:
        return a
    npad = [(0, 0) for x in range(axis_nb)]
    npad[axis] = (0, pad_size)
    return np.pad(a, pad_width=npad, mode="constant", constant_values=0)


if __name__ == '__main__':
    main()
