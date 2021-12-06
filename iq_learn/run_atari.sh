#!/usr/bin/env bash

# Atari experiments (Default: Use 20 expert demo)

# Pong
python train_iq.py agent=softq env=pong agent.init_temperature=1e-3 method.loss=value_mix method.chi=True seed=0

# Breakout
python train_iq.py agent=softq env=breakout agent.init_temperature=1e-3 method.loss=value_mix method.chi=True seed=0

# Space Invaders
python train_iq.py agent=softq env=space agent.init_temperature=1e-3 method.loss=value_mix method.chi=True seed=0 
