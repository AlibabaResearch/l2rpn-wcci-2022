# L2RPN WCCI 2022 competition

## Background
L2RPN (Learn To Run a Power Grid) is a series of competitions aimming at testing the potential of AI as power grid operators. This repository is for the 2nd winning solution of the L2RPN WCCI 2022 competition, which is themed "Energies of the Future and Carbon Neutrality". This competition introduces more renewable generators and storages than previous ones. It focuses on the challenge of uncertainty in electricity production and robustness of AI agents.

For more detail, please see the
[competition homepage (link)](https://codalab.lisn.upsaclay.fr/competitions/5410).

## Summary of our solution
In L2RPN WCCI 2022 competition, two categories of actions can be performed:
- Discrete actions, including connecting powerlines, switching substation busbars;
- Continuous actions, including generator redispatch, curtailment, and setting storage powers.

Our final submitted solution combines greedy search for discrete actions with optimization of QP problems for continuous actions. For discrete actions, the original action space has more than 70k actions for substation switch. To manage this, we follow [Qsinghua's PPO solution for L2RPN NeurIPS 2020](https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution) to reduce it to 314 actions. For continuous actions, we follow the [OptimCVXPY agent among available L2RPN baselines](https://github.com/rte-france/l2rpn-baselines/tree/master/l2rpn_baselines/OptimCVXPY).

We also explored reinforcement learning agents, but ended up with similar or mixed results. We are actively working on this and will share it when it surpasses the current solution with confidence. 

## Usage
The current solution does not require training. Please see submission/optimCVXPY directly. 

## Requirements
```
grid2op>=1.7.2
lightsim2grid>=0.7.0.post1
cvxpy>=1.2.1
```

## NOTICE
This repository is developed by Alibaba and licensed under the Mozilla Public License, v2.0.
It is developed based on third-party components under the same open source licenses. 
See the NOTICE file for more information.