# IQ-Learn

*NeurIPS '21 Spotlight Work* 

## Requirement

- tensorboardX
- pytorch (>= 1.4)
- gym
- wandb

## Installation

- Make a conda environment and install dependencies: `pip install -r requirements.txt`
- Download expert datasets from [Dropbox](https://www.dropbox.com/sh/xi92cwnrh0wqxa4/AABK9KFI-PxZ6fMaXJ2U8xKMa?dl=0)
- Setup wandb project to log and visualize metrics

## Instructions

Our training code is present in `train_iq.py` which implements IQ-Learn on top of DQN/SAC RL agents. The original RL training code is in `train_rl.py`.

- To reproduce our Offline IL experiments, see `run_offline.sh`
- To reproduce our Mujoco experiments, see `run_mujoco.sh`
- To reproduce Atari experiments, see `run_atari.sh`
- To visualize our recovered state-only rewards on a toy Point Maze environment: 
    `python -m vis.maze_vis env=pointmaze_right eval.policy=pointmaze agent.init_temperature=1 agent=sac double_q_critic._target_=agent.sac_models.DoubleQCritic`.
    Reward visualizations are saved in `vis/outputs` directory

## License

The code is made available for academic research use. Please see the [LICENSE](LICENSE.md) for the licensing terms for this code. 

For any inquiry, contact: Div Garg ([divgarg@stanford.edu](mailto:divgarg@stanford.edu?subject=[GitHub]%IQ-Learn))


