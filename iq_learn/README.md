# Inverse Q-Learning (IQ-Learn)

*IQ-Learn (NeurIPS '21 Spotlight):*  State-of-the-art framework for non-adversarial Imitation Learning.

## Requirement

- tensorboardX
- pytorch (>= 1.4)
- gym
- wandb
- hydra-core

## Installation

- Make a conda environment and install dependencies: `pip install -r requirements.txt`
- Setup wandb project to log and visualize metrics
- (Optional) Download expert datasets for Atari environments from [Dropbox](https://www.dropbox.com/sh/xi92cwnrh0wqxa4/AABK9KFI-PxZ6fMaXJ2U8xKMa?dl=0)

## Examples

We show some examples that push the boundaries of imitation learning using IQ-Learn:

### 1. CartPole-v1 using 1 demo subsampled 20 times with fully *offline* imitation  

```
python train_iq.py agent=softq method=iq env=cartpole  expert.demos=1 expert.subsample_freq=20 agent.init_temp=0.001 method.chi=True method.loss=value_expert
```

IQ-Learn is the only method thats reaches the expert env reward of **500** (requiring only 3k training steps and less than 30 secs!!)

<img src="../docs/cartpole_example.png" width="400"> 



## Instructions

Our training code is present in `train_iq.py` which implements **IQ-Learn** on top of DQN/SAC RL agents. <br> IQ-Learn simplify modifies the loss function for the critic network, compared to vanilla RL. The original RL training code is in `train_rl.py` for reference.

- To reproduce our Offline IL experiments, see `run_offline.sh`
- To reproduce our Mujoco experiments, see `run_mujoco.sh`
- To reproduce Atari experiments, see `run_atari.sh`
- To visualize our recovered state-only rewards on a toy Point Maze environment: 
    `python -m vis.maze_vis env=pointmaze_right eval.policy=pointmaze agent.init_temp=1 agent=sac double_q_critic._target_=agent.sac_models.DoubleQCritic`. <br>
    Reward visualizations are saved in `vis/outputs` directory

## License

The code is made available for academic research use. Please see the [LICENSE](LICENSE.md) for the licensing terms for this code. 

For any inquiry, contact: Div Garg ([divgarg@stanford.edu](mailto:divgarg@stanford.edu?subject=[GitHub]%IQ-Learn))


