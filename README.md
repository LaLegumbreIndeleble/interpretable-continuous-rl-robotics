# Interpretable Continuous Reinforcement Learning in Robotics

A research framework for training and analyzing interpretable policies in continuous action-space reinforcement learning for robotic systems.

## Overview

This project focuses on developing RL agents for continuous control tasks in robotics while maintaining **interpretability** — making it possible to understand *why* the policy takes certain actions, not just *what* actions it takes.

Key goals:
- Continuous action-space RL (e.g., PPO, SAC, TD3) applied to robotic tasks
- Interpretability methods: attention maps, saliency, symbolic regression on learned policies
- Support for standard robotic benchmarks (MuJoCo, IsaacGym, LeRobot, etc.)
- Modular design to swap environments, algorithms, and explanation methods

## Project Structure

```
interpretable-continuous-rl-robotics/
├── src/
│   ├── agents/         # RL algorithm implementations (PPO, SAC, TD3, ...)
│   ├── envs/           # Environment wrappers and custom envs
│   ├── models/         # Policy/value network architectures
│   └── utils/          # Logging, visualization, replay buffers
├── experiments/        # Experiment configs and run scripts
├── notebooks/          # Analysis and visualization notebooks
├── docs/               # Documentation
├── tests/              # Unit and integration tests
└── configs/            # Hydra/YAML configuration files
```

## Installation

```bash
git clone https://github.com/<your-username>/interpretable-continuous-rl-robotics.git
cd interpretable-continuous-rl-robotics
pip install -e .
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate icrl
pip install -e .
```

## Quick Start

```bash
# Train a SAC agent on HalfCheetah with interpretability logging
python experiments/train.py --config configs/sac_halfcheetah.yaml

# Analyze a trained policy
python experiments/analyze_policy.py --checkpoint runs/sac_halfcheetah/best.pt
```

## Interpretability Methods

| Method | Description | Status |
|--------|-------------|--------|
| Saliency Maps | Gradient-based input attribution | Planned |
| Attention Visualization | For transformer-based policies | Planned |
| Symbolic Regression | Distill policy into human-readable rules | Planned |
| Concept Bottleneck | Mid-level concept-based explanations | Planned |

## Environments

- [ ] MuJoCo (locomotion, manipulation)
- [ ] LeRobot (real-robot-ready tasks)
- [ ] IsaacGym / Isaac Lab (GPU-accelerated sim)
- [ ] Custom robotic tasks

## Algorithms

- [ ] SAC (Soft Actor-Critic)
- [ ] PPO (Proximal Policy Optimization)
- [ ] TD3 (Twin Delayed DDPG)
- [ ] Dreamer / MBPO (model-based)

## References

- [Soft Actor-Critic (Haarnoja et al., 2018)](https://arxiv.org/abs/1801.01290)
- [Proximal Policy Optimization (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [Interpretable RL Survey](https://arxiv.org/abs/2112.13112)
- [LeRobot](https://github.com/huggingface/lerobot)

## License

MIT
