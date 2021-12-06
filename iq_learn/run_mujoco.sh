#!/usr/bin/env bash

# Train on Mujoco environements (Default: Use 10 expert demo)
# Set eval.demos=1 for using one expert demo.

# Hopper-v2
python train_iq.py env=hopper agent=sac eval.demos=10 method.loss=v0 method.regularize=True agent.actor_lr=3e-5 seed=0

# HalfCheetah-v2
python train_iq.py env=cheetah agent=sac eval.demos=10 method.loss=value method.regularize=True agent.actor_lr=3e-05 seed=0

# Ant-v2
python train_iq.py env=ant agent=sac eval.demos=10 method.loss=value method.regularize=True agent.actor_lr=3e-05 agent.init_temperature=0.001 seed=0

# Walker2d-v2
python train_iq.py env=walker agent=sac eval.demos=10 method.loss=v0 method.regularize=True agent.actor_lr=3e-05 seed=0