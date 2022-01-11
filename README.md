# Inverse Q-Learning (IQ-Learn)
Official code base for **[IQ-Learn: Inverse soft-Q Learning for Imitation](https://arxiv.org/abs/2106.12142)**, ***NeurIPS '21 Spotlight***

**IQ-Learn** is the successor to Adversarial Imitation Learning methods like [GAIL](https://arxiv.org/abs/1606.03476) (coming from the same lab).\
It extends the theoretical framework for Inverse RL to non-adverserial and scalable learning, for the *first-time* showing guaranteed convergence.

[**[Project Page](https://div99.github.io/IQ-Learn)**]

Update: **IQ-Learn** was recently used to create the **best AI agent** for playing Minecraft. Placing **#1** in NeurIPS MineRL Basalt Challenge using only human demos (Overall Leaderboard Rank **#2**)
### Citation
```
@inproceedings{garg2021iqlearn,
title={IQ-Learn: Inverse soft-Q Learning for Imitation},
author={Divyansh Garg and Shuvam Chakraborty and Chris Cundy and Jiaming Song and Stefano Ermon},
booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
year={2021},
url={https://openreview.net/forum?id=Aeo-xqtb5p}
}
```

## Introduction

We introduce **Inverse Q-Learning (IQ-Learn)**, a state-of-art novel framework for imitation learning, that directly learns **soft-Q** functions from expert data. IQ-Learn enables *non-adverserial* imitation learning (IL), working on both offline and online IL settings. It is performant even with *very sparse* expert data, and scales to complex image-based environments, surpassing prior methods by more than **3x**.

Inverse Q-Learning is theoretically equivalent to *Inverse Reinforcement learning*, i.e. learning rewards from expert data. However, it is much more powerful in practice. It admits very simple non-adverserial training and works on complete offline IL settings (without any access to the environment).

## Key Advantages

✅  Non-adverserial online IL (Successor to [GAIL](https://arxiv.org/abs/1606.03476) & [AIRL](https://arxiv.org/abs/1710.11248)) \
✅  Simple to implement  \
✅  Performant with very sparse data (single expert demo) \
✅  Scales to Complex  Image Envs (SOTA on Atari and playing Minecraft) \
✅  Recover rewards from envs


## Imitation 
**Reaching human-level performance on Atari with pure imitation:**

<p float="left">
<img src="videos/pong.gif" width="250">
<img src="videos/breakout.gif" width="250">
</p>
<p float="left">
<img src="videos/space.gif" width="250">
<img src="videos/qbert.gif" width="250">
</p>

## Rewards
Recovering environment rewards on GridWorld:

![Grid](videos/grid.jpg)



## Questions
Please feel free to email us if you have any questions. 

Div Garg ([divgarg@stanford.edu](mailto:divgarg@stanford.edu?subject=[GitHub]%IQ-Learn))
