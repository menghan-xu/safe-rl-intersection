# Uncertainty-Aware Collision Avoidance through Safe Reinforcement Learning

This project addresses the challenge of decision-making for autonomous vehicles navigating unsignalized intersections under uncertainty. Our primary objective is to enable an ego vehicle to safely cross a 3m x 3m intersection while avoiding collisions with a human-driven agent vehicle that follows probabilistic trajectories.

Unlike traditional planning methods that often struggle to balance safety guarantees with efficiency, we propose an uncertainty-aware Safe Reinforcement Learning (RL) framework formulated as a Constrained Markov Decision Process (CMDP). We utilize Lagrange Proximal Policy Optimization (LPPO) to dynamically balance task rewards against hard safety constraints.

## Setup

### 1. Install packages
We recommend using a conda environment. This project is tested on Python 3.11.11.

```bash
pip install -r requirements.txt
```
- Note: If you are on Windows/Linux with an NVIDIA GPU, you may need to reinstall PyTorch with CUDA support.

- Mac Users: The training script should indicate "Running on mps".

- Other: If no GPU is detected, it will default to "Running on cpu".
### 2. Data Setup
Create a folder named data at the same level as the code folder. Place the expert_agent_trajs.npy file inside this folder.

Your directory structure should look like this:
```text
project_root/
├── code/
│   ├── lagrange_ppo/
│   └── ... (other methods)
└── data/
    └── expert_agent_trajs.npy
```

### 3. Training
You can tune hyperparameters in the hyperparameters.yaml file located within each method's specific folder (e.g., code/lagrange_ppo/hyperparameters.yaml).

To start training:
```bash
python train.py
```

Output: During training, a folder will be created at code/{method_name}/learned_policies/Intersection-{Method}-v0_{timestamp} containing:

- config.yaml: A snapshot of the hyperparameters.

- model_checkpoints: Saved model weights.

- gifs: Visualizations of training progress (iter_x.gif) and testing episodes (test_ep_x.gif).

- report.txt: Summary of final metrics (Success Rate, Collision Rate, Average Reward).

## Problem Formulation
We formulate the autonomous navigation task as a Constrained Markov Decision Process (CMDP), defined by the tuple $(S, A, P, R, C, \gamma)$. 

### State and Action Space
The state $s_t$ captures the ego vehicle's longitudinal kinematics and the agent's observable state, including perceptual uncertainty. The action $a_t$ is the continuous longitudinal acceleration.

```math
s_t = [y_{ego}, v_{ego}, x_{agent}, y_{agent}, v_{x, agent}, v_{y, agent}, \sigma_{x,agent}, \sigma_{y,agent}]^\top
```

### Reward Function
The reward function is designed to encourage efficient navigation while ensuring passenger comfort and adherence to traffic regulations.

$$ r(s_t, a_t) = r_{nav} + r_{comfort} + r_{sparse}$$

Where the components are defined as:

```math
\begin{align}
r_{nav} &= w_{prog}(v_{ego}\Delta t) - w_{time}\Delta t - w_{speed}\max(0, v_{ego} - v_{limit})^2 \\
r_{comfort} &= -w_{comf} \cdot a_t^2 \\
r_{sparse} &= \begin{cases} R_{goal}, & \text{if } y_{ego} \ge y_{target} \\ -C_{crash}, & \text{if collision} \\ 0, & \text{otherwise} \end{cases}
\end{align}
```

### Safety Cost Function

To ensure safety under perceptual uncertainty, we define a cost function based on a conservative safety boundary. Let $R_{ego}$ and $R_{agent}$ be the collision radii of the vehicles.

The conservative safety threshold $d_{cons}$ incorporates the agent's positional uncertainty:
```math
d_{cons} = (R_{ego} + R_{agent} + \sqrt{\sigma_{x,agent}^2 + \sigma_{y,agent}^2})^2
```

The safety cost $c(s_t)$ imposes a continuous penalty when the actual distance breaches this boundary:
```math
c(s_t) = w_{safe} \cdot \max(0, d_{cons} - d_{actual}) + c_{collision}
```

## Methodologies

This repository implements and compares five distinct approaches:

### 1. Lagrange PPO (LPPO)

This is the core framework. We utilize Lagrangian relaxation to transform the constrained optimization problem into an unconstrained dual problem.

* **Mechanism**: Uses a dual-head Critic to estimate Task Reward and Safety Cost separately.
* **Update**: Optimizes a combined Lagrangian advantage: $A^{Lag} = A^R - \lambda \cdot A^C$.
* **Adaptive Constraints**: The Lagrange multiplier $\lambda$ automatically increases if the safety constraint is violated, forcing the policy to prioritize safety over speed.

### 2. LPPO with Behavior Cloning (LPPO+BC)

To enhance sample efficiency and ensure realistic driving behavior, we integrate Behavior Cloning (BC) as an auxiliary supervised learning objective.

* **Objective**: Adds a Mean Squared Error (MSE) loss between the policy's action and expert demonstrations during updates.
* **Benefit**: Significantly improves training stability and navigational efficiency.

### 3. LPPO with Barrier Force Function (LPPO+BFF)

Integrates a physics-informed safety filter.

* **Mechanism**: Decomposes the control action into a nominal policy output and a repulsive force derived from logarithmic barrier functions.
* **Benefit**: Provides proactive safety filtering during training.

### 4. Non-Linear MPC

A deterministic baseline utilizing the CasADi framework and IPOPT solver.

* **Formulation**: Optimizes a finite-horizon trajectory subject to hard safety constraints based on the conservative uncertainty boundary.

### 5. Residual MPC (LPPO+MPC)

A hybrid architecture combining model-based control with reinforcement learning.

* **Structure**: The MPC provides a nominal action, and the RL agent learns a residual correction term to compensate for model mismatches or solver limitations.
```math
a_{final} = \text{clip}(a_{MPC} + w_{res} \cdot a_{res}, -a_{max}, a_{max})
```

## Results

Experiments in high-fidelity simulation and on hardware (Clearpath Jackal robots) demonstrate that our LPPO-based policies achieve a 100% success rate across diverse scenarios, including zero-shot generalization to out-of-distribution agent behaviors.


For more details, please refer to our final report:

**[Uncertainty-Aware Collision Avoidance via Safe RL](Uncertainty%20Aware%20Collision%20Avoidance%20through%20Safe%20Reinforcement%20Learning.pdf)**
