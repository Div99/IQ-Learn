#!/usr/bin/env bash

# Offline Imitation learning experiments (Default: Use 10 expert demo)
# Set eval.demos=1 for using one expert demo, reduce train.batch if the expert buffer is too small.

# Acrobot-v1
python train_iq.py agent=softq env=acrobot eval.demos=10 eval.subsample_freq=5 method.chi=True method.loss=value_expert seed=0

# CartPole-v1
python train_iq.py agent=softq env=cartpole eval.demos=10 eval.subsample_freq=20 method.chi=True method.loss=value_expert agent.init_temperature=0.001 seed=0

# LunarLander-v2
python train_iq.py agent=softq env=lunarlander eval.demos=10 eval.subsample_freq=5 method.chi=True method.loss=value_expert agent.init_temperature=0.001 seed=0