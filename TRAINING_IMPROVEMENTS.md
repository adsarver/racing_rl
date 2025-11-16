# F1 Racing RL Training Improvements

## ✅ Implemented Changes

The following improvements have been successfully implemented:

1. **Fixed Reward Shaping** - Balanced reward scales (progress, collision, slide penalties)
2. **Increased Entropy Coefficient** - From 0.01 → 0.05 to maintain exploration
3. **Increased Clip Epsilon** - From 0.1 → 0.2 for better policy updates
4. **Increased Learning Rate** - From 5e-5 → 3e-4 for faster convergence
5. **Extended Map Training Duration** - From 10 → 30 generations per map
7. **Adaptive Entropy Coefficient** - Automatic adjustment to prevent exploration collapse
9. **Track Boundary Awareness Reward** - Penalty for getting too close to walls

---

## 🔄 Remaining Recommendations

### 1. **Curriculum Learning - Map Difficulty Scheduler**

Gradually introduce more challenging tracks as training progresses.

**Map Analysis** (✅ Verified - all 18 maps have compatible raceline data):
- **Easy:** Austin, Melbourne, Hockenheim, Monza
- **Medium:** BrandsHatch, Budapest, Catalunya, Spa, Silverstone, Zandvoort
- **Hard:** MoscowRaceway, Nuerburgring, Oschersleben, Sakhir, Sepang, YasMarina, SaoPaulo, Sochi

**Implementation - Add to train.py:**

```python
# Curriculum Learning - Maps organized by difficulty (add after MAP_NAMES)
EASY_MAPS = ["Austin", "Melbourne", "Hockenheim", "Monza"]
MEDIUM_MAPS = ["BrandsHatch", "Budapest", "Catalunya", "Spa", "Silverstone", "Zandvoort"]
HARD_MAPS = ["MoscowRaceway", "Nuerburgring", "Oschersleben", "Sakhir", "Sepang", "YasMarina", "SaoPaulo", "Sochi"]

def get_curriculum_map_pool(generation):
    """Returns the appropriate map pool based on training progress."""
    if generation < 100:
        return EASY_MAPS  # Phase 1: Learn basics
    elif generation < 300:
        return EASY_MAPS + MEDIUM_MAPS  # Phase 2: Add complexity
    else:
        return EASY_MAPS + MEDIUM_MAPS + HARD_MAPS  # Phase 3: Full curriculum

# Replace the existing map change logic with:
if (gen+1) % GEN_PER_MAP == 0:
    available_maps = get_curriculum_map_pool(gen+1)
    CURRENT_MAP = random.choice(available_maps)
    print(f"Gen {gen+1}: Map={CURRENT_MAP}, Pool size={len(available_maps)}")
    
    INITIAL_POSES = generate_start_poses(CURRENT_MAP, NUM_AGENTS)
    env.update_map(get_map_dir(CURRENT_MAP) + f"/{CURRENT_MAP}_map", ".png")
    
    agent.waypoints_xy, agent.waypoints_s, agent.raceline_length = agent._load_waypoints(CURRENT_MAP)
    agent.last_cumulative_distance = np.zeros(agent.num_agents) 
    agent.last_wp_index = np.zeros(agent.num_agents, dtype=np.int32)
    
    env.reset(poses=INITIAL_POSES)
    agent.reset_progress_trackers(initial_poses_xy=INITIAL_POSES[:, :2])
```

---

### 2. **Learning Rate Scheduling**

```python
# In PPOAgent.__init__, add:
from torch.optim.lr_scheduler import CosineAnnealingLR

self.actor_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=num_generations, eta_min=1e-6)
self.critic_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=num_generations, eta_min=1e-6)

# In learn(), after optimizer.step():
self.actor_scheduler.step()
self.critic_scheduler.step()
```

---

### 3. **Value Network Warm-up**

```python
# In PPOAgent.learn(), adjust epochs dynamically:
if self.generation_counter < 50:
    critic_epochs = 8
    actor_epochs = 5
else:
    critic_epochs = 5
    actor_epochs = 5
```

---

### 4. **Speed Target Reward**

```python
# In calculate_reward():
optimal_speed = 15.0
current_speed = next_obs['linear_vels_x'][i]
speed_error = abs(current_speed - optimal_speed) / optimal_speed
reward += -0.2 * speed_error
```

---

## Monitoring Tips

✅ **Healthy Training:**
- Entropy > -2.0
- KL divergence < 0.1
- Clip fraction 0.1-0.3
- Rewards increasing
- Collisions decreasing

❌ **Warning Signs:**
- Entropy < -4.0
- KL divergence > 0.2
- Rewards plateau 50+ gens
- Critic loss spiking

---

## Design Decisions

✅ Minimal state (LIDAR + odometry only)
✅ No waypoint features (for generalization)
✅ All 18 maps verified with compatible data
