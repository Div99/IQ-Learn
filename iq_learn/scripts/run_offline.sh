#!/usr/bin/env bash

# Offline Imitation learning experiments (Default: Use 10 expert demo)
# Set expert.demos=1 for using one expert demo, reduce train.batch if the expert buffer is too small.

# Set working directory to iq_learn
cd ..

# Acrobot-v1
python train_iq.py agent=softq env=acrobot expert.demos=10 expert.subsample_freq=5 method.chi=True method.loss=value_expert seed=0

# CartPole-v1
python train_iq.py agent=softq env=cartpole expert.demos=10 expert.subsample_freq=20 method.chi=True method.loss=value_expert agent.init_temp=0.001 seed=0

# LunarLander-v2
python train_iq.py agent=softq env=lunarlander expert.demos=10 expert.subsample_freq=5 method.chi=True method.loss=value_expert agent.init_temp=0.001 seed=0
