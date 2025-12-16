# Barrier Force Function (BFF) Integration

## Overview
This implementation integrates the **Barrier Force Function** method from the paper "Model-Based Safe Reinforcement Learning with Time-Varying State and Control Constraints" into your Lagrange PPO model.

## What Changed

### New Files
1. **`barrier_functions.py`** - Implements barrier functions for safety:
   - `SimpleCollisionBarrier` - Prevents collisions with dynamic obstacles
   - `SimpleControlBarrier` - Enforces acceleration limits

### Modified Files
2. **`models.py`** - Added BFF parameters:
   - `use_barriers` flag (default: True)
   - `rho = 0.1` - Control barrier gain
   - `K = 0.05` - State barrier gain

3. **`train.py`** - Integrated barrier forces into rollout:
   - Initializes barrier functions
   - Applies barrier forces during action selection
   - Formula: `u = v + ρ*∇B(v) + K*∇B(x)`

## How It Works

### Mathematical Basis
The BFF modifies how actions are generated:

**Before BFF:**
```
action = policy(state)  # Direct output
```

**With BFF:**
```
virtual_action = policy(state)  # "v" in the paper
collision_force = state_barrier.gradient(ego_pos, agent_pos)
control_force = control_barrier.gradient(virtual_action)
action = virtual_action + ρ*control_force + K*collision_force
```

### Physical Interpretation
- **Collision force**: Pushes ego vehicle away from obstacles (repulsive potential)
- **Control force**: Keeps acceleration within safe limits
- Think of barriers as "invisible walls" that gently push the agent to safety

### Key Parameters
- **`rho`** (0.1): How strongly to enforce control limits
- **`K`** (0.05): How strongly to enforce collision avoidance
- **`epsilon`** (0.1): Relaxation factor for numerical stability

## Usage

### Quick Test (5 minutes)
```bash
cd /home/doris/Github/safe-rl-intersection/code/lagrange_ppo
python test_bff.py
```

This runs 50 training epochs to verify the integration works.

### Full Training
```bash
python train.py
```

Uses your existing `hyperparameters.yaml` configuration.

### Disable BFF (if needed)
In `models.py`, change:
```python
model = ContinuousActorCritic(8, 1, cfg['max_accel'], use_barriers=False)
```

## Expected Results

### Safety Improvements
- ✓ Fewer collisions during training
- ✓ Smoother exploration (no extreme accelerations)
- ✓ Better constraint satisfaction

### Performance
- May slightly reduce peak reward (more conservative)
- Should improve success rate and robustness
- Potentially faster convergence due to safer exploration

## Integration with Lagrangian
BFF and Lagrangian multipliers work **complementarily**:
- **BFF (Proactive)**: Prevents constraint violations during exploration
- **Lagrangian (Reactive)**: Penalizes violations after they occur

## Troubleshooting

### Issue: Training crashes
**Solution**: Check `epsilon` parameter in barrier functions. Increase if getting numerical instabilities.

### Issue: Too conservative (low reward)
**Solution**: Reduce `rho` and `K` values in `models.py`:
```python
self.rho = 0.05  # Lower = less conservative
self.K = 0.02    # Lower = less conservative
```

### Issue: Still getting collisions
**Solution**: Increase `K` value:
```python
self.K = 0.1  # Higher = more collision avoidance force
```

## Next Steps

### Stage 1 (Current) ✓
- BFF integrated into policy structure
- Fixed barrier gains (rho, K)
- Reward/cost structure unchanged

### Stage 2 (Future - Optional)
- Learn rho and K as trainable parameters
- Add barrier terms to cost function
- Implement multi-step policy evaluation (MPE)

## Technical Details

### Barrier Gradient Computation
For collision avoidance:
```
G(x) = r_safe² - d²  (constraint function)
∇B(x) = -∇G(x) / G   (barrier gradient)
```

Where:
- `r_safe` = collision_radius + uncertainty
- `d` = distance to obstacle
- Force is large when close to constraint (G → 0)

### Numerical Stability
- Relaxation factor `epsilon` prevents division by zero
- Gradient clamping prevents explosion
- Only applies force near constraints (efficient)

## References
- Paper: "Model-Based Safe Reinforcement Learning with Time-Varying State and Control Constraints"
- Key equation: Equation (13) - Barrier Force-Based Control Policy
- Method: Equation (15) - Multi-Step Policy Evaluation (not implemented yet)

## Questions?
Check the plan document for detailed implementation notes and troubleshooting.
