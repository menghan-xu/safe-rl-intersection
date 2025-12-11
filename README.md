# Updates 12-09

(1) Changed the data processing files according to new expert data, with the processed data shown in the `data` folder.

(2) Saved the best training policy with the largest reward (you guys can also try lowest loss instead).

(3) Included the best results I have so far in the folder `learned_policies`.


# Safe Reinforcement Learning for Intersection Navigation

This project addresses the challenge of decision-making for autonomous vehicles navigating unsignalized intersections under uncertainty. Our primary objective is to enable an ego vehicle to safely cross a $3m \times 3m$ intersection while avoiding collisions with a human-driven agent vehicle that follows probabilistic trajectories.

Initially, we formulated this control problem using standard Proximal Policy Optimization (PPO) to maximize a unified reward function. However, standard PPO formulations maximize expected returns without explicitly incorporating auxiliary safety constraints or distinct cost signals. To rigorously address safety requirements and maintain a strict separation between task performance and collision avoidance, we transitioned to a Lagrangian PPO framework. This approach treats safety violations as a distinct cost signal constrained by an adaptive Lagrange multiplier, effectively separating performance optimization from safety enforcement.

## Lagrangian PPO
We model the interaction as a Constrained Markov Decision Process (CMDP) defined by the tuple $(S, A, P, R, C, \gamma)$. The dynamics follow a linear point-mass model.

**State and Action Space** The state $s_t \in \mathbb{R}^8$ concatenates the ego vehicle's longitudinal state with the agent's observed state and uncertainty. The action $a_t \in \mathbb{R}$ represents the continuous longitudinal acceleration of the ego vehicle.
$$s_t = [y_{ego}, v_{ego}, x_{agent}, y_{agent}, v_{x, agent}, v_{y, agent}, \sigma_{x,agent}, \sigma_{y,agent}]^\top$$
$$a_t \in [a_{min}, a_{max}]$$

The ego is only going forward for now, so then $x_{ego}$ is fixed as $0.5$. (maybe expore 2-d problem later).


**Reward Function** To encourage efficient navigation while maintaining passenger comfort and adhering to traffic rules, the reward function $r_t$ is composed of progress incentives, speed regulation, and control penalties.
$$r(s_t, a_t) = r_{\text{progress}} + r_{\text{overspeed}} + r_{\text{comfort}} + r_{\text{terminal}} + r_{\text{collision}}$$


Where:
$$r_{\text{progress}} = \alpha \cdot v_{ego} \cdot \Delta t$$

$\Delta t = 0.1s$

$$r_{\text{overspeed}} = -\beta \cdot \max(0, v_{ego} - v_{limit})^2$$
$$r_{\text{comfort}} = -\eta \cdot a_t^2$$
Considering the constraints of the Jackal robot, we set $v_{\text{limit}} = 1.5\,\text{m/s}$.

$$r_{terminal} =
\begin{cases}
    50, &\text{if }y_{ego} = y_{target}\\
    0, &\text{otherwise}
\end{cases}
$$


In our setting, $y_{target}=1.5$

$$r_{collison} =
\begin{cases}
    -100, & \text{if } d_{\text{safety}} > d_{\text{actual}} \\
0, & \text{otherwise}
\end{cases}
$$

Add time penalty to the ego to push it reach target sooner if possible.
$r_{time} = -w_{time} \cdot \Delta t$
**Cost Function (Safety)** 
<!-- Safety is quantified using the Mahalanobis distance, which accounts for the probabilistic nature of the agent's position. We define the squared Mahalanobis distance $D^2$ between the ego position $u_t$ and the agent's distribution $\mathcal{N}(\mu_t, \Sigma_t)$ as:$$D^2(u_t, \mu_t) = (u_t - \mu_t)^\top \Sigma_t^{-1} (u_t - \mu_t)$$The safety cost $c_t$ is formulated as an exponential barrier function bounded between $[0, 100]$, ensuring a smooth gradient as the risk increases:$$c(s_t, a_t) = 100 \cdot \exp \left( -\frac{D^2(u_t, \mu_t)}{2} \right)$$ -->
$R_{ego}$ is the half-diagonal of the ego, and \(R_{agent}\) is the half-diagonal of the agent. In our setting, the Jackal robot’s half-diagonal is $0.3328\,\text{m}$.
$$d_{actual} = (x_{ego}-x_{agent})^2+(y_{ego} - y_{agent})^2$$
$$d_{safety} = (R_{ego} + R_{agent}  )^2$$
$$d_{conservative} = (R_{ego} + R_{agent} +\sqrt{\sigma_{x, agent}^2 + \sigma_{y,agent}^2})^2$$
$$c(s_t) = \mu\max(0, d_{conservative} - d_{actual}) +c_{collision}$$

$$
c_{\text{collision}} = 
\begin{cases}
100, & \text{if } d_{\text{safety}} > d_{\text{actual}} \\
0, & \text{otherwise}
\end{cases}
$$

**Lagrangian Optimization Objective** We aim to maximize the expected return subject to a safety cost limit $d$. We define the per-step Lagrangian advantage $A^{Lag}$ by combining the reward advantage $A^R$ and the cost advantage $A^C$ via the Lagrange multiplier $\lambda$:

$$A^{Lag}_t = A^R_t - \lambda A^C_t$$

The policy $\pi_\theta$ is updated by maximizing the PPO surrogate objective using $A^{Lag}_t$. 

$$l_{\pi}(\theta) = -\sum_{s,a}\min (\frac{\pi_{\theta}(a|s)}{\pi_{\theta_t}(a|s)}A_t^{Lag},\text{clip}(\frac{\pi_{\theta}(a|s)}{\pi_{\theta_t}(a|s)}, 1-\epsilon, 1+\epsilon)A_t^{Lag} )$$

Simultaneously, the Lagrange multiplier $\lambda$ is updated via dual gradient ascent to satisfy the safety constraint:
$$\lambda_{k+1} = \max \left( 0, \lambda_k + \eta_{\lambda} (J_C(\pi_\theta) - d) \right)$$
where $J_C(\pi_\theta)$ is the expected safety cost of the current policy, and $d$ is the safety budget. This mechanism adaptively increases the penalty weight when the agent violates the safety threshold and decreases it when the behavior is safe.


### Hyperparameters Tuning
dt : 0.1
max_accel = 3.2
Alpha (w_progress) : 17.11
Eta (w_comfort)    : 0.697
Beta (w_overspeed) : 72.7
Mu (cost_scale)    : 463.2

### Set up


**Install packages**
I personaly prefer conda environment. My python version is 3.11.11
```bash
pip install -r requirements.txt
```
If you are on Windows/Linux with an NVIDIA GPU, you might need to reinstall PyTorch with CUDA support (the requirements file installs the CPU version)

If the environment is set up correctly, running `train.py` should show “Running on cuda.”  
If you're on a Mac, you'll see “Running on mps,” and if you're on a CPU-only setup, it will show “Running on cpu.”


**Put the `.npy` file in the correct folder**

Create a folder called `data` at the same level as your `code` folder, and place the `expert_agent_trajs.npy` file inside it. I've uploaded this data file to our shared google drive.

Your directory structure should look like this:
```
project_root/
├── code/
│   ├── lagrange_ppo/
│   └── ppo/
└── data/
    └── expert_agent_trajs.npy
```

**Tune hyperparameters in `lagrange_ppo/hyperparameters.yaml`**
I have already written explanations for every hyperparameter, and some guidelines on how to tune them. After you change hyperparameters in this file, run 
```
python train.py
```
to start training, and then testing.

For Mac, training is expected to take around 30 minutes. For CUDA, I don’t know the exact time, but it should be faster depending on your GPU.

During training, a folder will be created at  
`code/lagrange_ppo/learned_policies/Intersection-Lag-v0_{your_time}`

This folder contains:
- `config.yaml`: your hyperparameter settings  
- reward and loss curves  
- model checkpoints  
- `iter_{number}.gif`: animation during training  
- `test_ep_{number}.gif`: animation during testing  
- `report.txt`: summary of the final testing results  
  - `Avg Reward`     
  - `Success Rate`    
  - `Collision Rate`  

Feel free to message me if you have any questions!

## Lagrangian PPO with Behavior Cloning
Given we have a well-collected expert dataset, we want the model to leverage expert demonstrations by adding a behavior cloning loss as an auxiliary objective, so the policy can anchor itself to expert behavior.

At the same time, we still want the agent to explore new possibilities and avoid distribution shift, while treating expert actions as a reference rather than a hard constraint.

$$\mathcal{l}_{BC}(\theta) = \frac{1}{N} \sum_{i=1}^{N} || \mu_\theta(s_i) - a_i^* ||^2_2$$

$\mu_\theta(s_i)$ is the action we chose given the current policy $\pi_{\theta}$

Adding the behavior cloning loss to the total loss with a multiplier $w_{BC}$, we regularize the policy towards expert behavior while still optimizing the RL objective.

## Some other new ideas
1. Physics-Informed Residual Architecture
Introduce a hybrid control scheme where actions are decomposed into $a_t = a_{\text{LQR}}(s_t) + a_{\theta}(s_t)$. The RL agent only learns the residual adjustment ($a_{\theta}$) needed for collision avoidance, while the LQR controller handles vehicle kinematics and stability.
