import os
import sys
import numpy as np
import torch

# MARL ray imports
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog

# Import local models
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from attention_model_A import AttentionSACModel
from bluesky_gym.envs.ma_env_two_stage_AM_PPO_NOISE_kalman import SectorEnv
from ray.tune.registry import register_env
from attention_visualization import plot_searchlight

# --- Setup Configuration ---
RUN_ID = "attn_viz_run"
N_AGENTS = 20
CHECKPOINT_DIR = os.path.join(script_dir, "models/sectorcr_ma_sac/best_iter_00118")

def main():
    # 1. Initialize Ray and Models
    ray.shutdown()
    ray.init(include_dashboard=False)
    
    register_env("sector_env", lambda config: SectorEnv(**config))
    ModelCatalog.register_custom_model("attention_sac", AttentionSACModel)
    
    # 2. Build Policy
    cfg = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment("sector_env", env_config={"n_agents": N_AGENTS, "run_id": RUN_ID}, disable_env_checking=True)
        .framework("torch")
        .env_runners(num_env_runners=0)
        .training(
            model={
                "custom_model": "attention_sac",
                "custom_model_config": {
                    "hidden_dims": [512, 512],
                    "is_critic": False,
                    "n_agents": N_AGENTS,
                    "embed_dim": 128,
                },
                "free_log_std": True,
                "vf_share_layers": False,
            }
        )
        .multi_agent(
            policies={"shared_policy": (None, None, None, {})},
            policy_mapping_fn=lambda *_, **__: "shared_policy",
        )
    )
    
    algo = cfg.build()
    policy = algo.get_policy("shared_policy")
    
    # Load model weights
    policy_state_path = os.path.join(CHECKPOINT_DIR, "policies", "shared_policy", "policy_state.pkl")
    import pickle
    with open(policy_state_path, "rb") as f:
        policy_state = pickle.load(f)
    if "weights" in policy_state:
        model_weights_converted = {
            k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v 
            for k, v in policy_state["weights"].items()
        }
        policy.model.load_state_dict(model_weights_converted, strict=False)
        print("Model Weights Loaded!")

    # 3. Create Environment
    env = SectorEnv(render_mode=None, n_agents=N_AGENTS, run_id=RUN_ID)
    obs, _ = env.reset()
    
    print("\nSimulating steps to collect proper attention states...")
    
    figures_dir = os.path.join(script_dir, "figures_attention")
    os.makedirs(figures_dir, exist_ok=True)
    
    for step in range(1, 100):
        # We process step
        agent_ids = list(obs.keys())
        obs_array = np.stack(list(obs.values()))
            
        # Calculate Actions + get Model internal state
        actions_out, _, _ = policy.compute_actions(obs_array, explore=False)
        
        # Run forward manually JUST to scoop out the attention weights
        with torch.no_grad():
            input_dict = {"obs": torch.from_numpy(obs_array).float()}
            _, _ = policy.model(input_dict, [], None)
            
            # The model tracks weights here inside:
            attn_weights_all = policy.model._last_attn_weights 
            # if 3D (Batch, heads, N) or (Batch, N)
            if len(attn_weights_all.shape) == 3:
                attn_weights_all = attn_weights_all.mean(axis=1) # mean over heads
        
        # Plot the figure for a few consecutive steps
        if 2 <= step <= 20 and len(agent_ids) > 0:
            target_agent = agent_ids[0]  # Grab the first available plane (Like KL001)
            agent_idx = 0

            
            # 1. Parse Ownship Position
            obs_vec = obs_array[agent_idx]
            # Normalization constants used in your environment
            MAX_X, MAX_Y = 8500.0, 8000.0
            
            own_x_km = (obs_vec[3] * MAX_X) / 1000.0
            own_y_km = (obs_vec[4] * MAX_Y) / 1000.0
            ownship_pos = (own_x_km, own_y_km)
            
            own_vx = obs_vec[5] * 36.0
            own_vy = obs_vec[6] * 36.0
            ownship_vel = (own_vx, own_vy)
            
            neigh_pos = {}
            neigh_vel = {}
            attn_weights_dict = {}
            
            num_intruders = (len(obs_vec) - 7) // 5
            
            for i in range(num_intruders):
                start_ptr = 7 + i * 5
                
                # Check for zero pad
                if np.all(obs_vec[start_ptr:start_ptr+5] == 0):
                    continue
                
                # In your obs: [dxi, dyi, dvx, dvy] are indices 1 and 2
                dxi_norm = obs_vec[start_ptr + 1]
                dyi_norm = obs_vec[start_ptr + 2]
                dvx_norm = obs_vec[start_ptr + 3]
                dvy_norm = obs_vec[start_ptr + 4]
                
                int_x_km = own_x_km + (dxi_norm * MAX_X) / 1000.0
                int_y_km = own_y_km + (dyi_norm * MAX_Y) / 1000.0
                
                int_vx = own_vx + (dvx_norm * 36.0)
                int_vy = own_vy + (dvy_norm * 36.0)
                
                # Let's map true ID from Environment if active
                if (i+1) < len(agent_ids):
                    target_id = agent_ids[i+1] # e.g. KL002
                else:
                    target_id = f"Plane_Hidden_{i}"
                
                neigh_pos[target_id] = (int_x_km, int_y_km)
                neigh_vel[target_id] = (int_vx, int_vy)
                weight = float(attn_weights_all[agent_idx, i])
                attn_weights_dict[target_id] = weight
            
            save_dest = os.path.join(figures_dir, f"attention_mechanism_{target_agent}_step{step}.png")
            
            # Use your visualization file (only searchlight)
            plot_searchlight(
                ownship_pos=ownship_pos,
                neigh_pos=neigh_pos,
                attn_weights=attn_weights_dict,
                ownship_vel=ownship_vel,
                neigh_vel=neigh_vel,
                ownship_id=target_agent,
                step=step,
                save_path=save_dest
            )
            print(f"\n✅ Visualisation successfully saved to -> {save_dest}")
            if step >= 13:
                break # Finished after drawing the plots!
                
        # Step env forward
        actions = {agent_id: action for agent_id, action in zip(agent_ids, actions_out)}
        obs, rewards, terminateds, truncateds, infos = env.step(actions)

    ray.shutdown()
    env.close()

if __name__ == "__main__":
    main()
