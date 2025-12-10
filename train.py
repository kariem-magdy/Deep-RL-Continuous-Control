import gymnasium as gym
import torch
import numpy as np
import os
import wandb
import argparse
import time
import glob
from agents.SAC import SACAgent
from agents.TD3 import TD3Agent
from agents.PPO import PPOAgent

# Ensure directories exist
os.makedirs("saved_models", exist_ok=True)
os.makedirs("videos", exist_ok=True)

def make_env(env_name, render_mode=None):
    if env_name in ["LunarLander-v3", "CarRacing-v3"]:
        return gym.make(env_name, continuous=True, render_mode=render_mode)
    return gym.make(env_name, render_mode=render_mode)

def evaluate_policy(agent, env_name, model_type, max_action, step_count):
    """
    Runs an evaluation episode.
    - Records video
    - Measures duration
    - Logs to WandB
    """
    video_folder = f"videos/{model_type}_{env_name}/step_{step_count}"
    
    # Create eval env with video recording
    eval_env = make_env(env_name, render_mode='rgb_array')
    eval_env = gym.wrappers.RecordVideo(
        eval_env, 
        video_folder=video_folder,
        disable_logger=True
    )
    
    state, _ = eval_env.reset()
    done = False
    total_reward = 0
    start_time = time.time()
    
    while not done:
        # Select action without exploration noise (evaluate=True)
        # For PPO, this uses the deterministic mean
        action = agent.select_action(state, evaluate=True)
        state, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        total_reward += reward
        
    duration = time.time() - start_time
    eval_env.close()
    
    # Robustly find the video file (handles variable naming by Gym)
    mp4_files = glob.glob(f"{video_folder}/*.mp4")
    video_path = mp4_files[0] if mp4_files else None

    # Log metrics
    log_dict = {
        "test/reward": total_reward,
        "test/episode_duration": duration
    }
    
    if video_path:
        log_dict["test/video"] = wandb.Video(video_path, fps=30, format="mp4")
        
    wandb.log(log_dict)
    print(f"--> Evaluation at step {step_count}: Reward {total_reward:.2f}, Duration {duration:.2f}s")
    return total_reward

def train(model_type, env_name, timesteps=None):
    # Determine timesteps: Use argument if provided, else default based on env
    if timesteps is None:
        timesteps = 200000 if env_name == "CarRacing-v3" else 100000

    config = {
        "learning_rate": 3e-4,
        "batch_size": 128,
        "buffer_size": 100000,
        "gamma": 0.99,
        "total_timesteps": timesteps,
        "eval_freq": 10000 if env_name == "CarRacing-v3" else 5000,
        "model": model_type,
        "env": env_name
    }
    
    run_name = f"{model_type}_{env_name}_Run"
    wandb.init(project="RL_Assignment4", name=run_name, config=config, reinit=True)
    
    env = make_env(env_name)
    
    # Check for Image Environment
    use_cnn = False
    if len(env.observation_space.shape) == 3: # (96, 96, 3)
        print(f"Detected Image Environment: {env_name}")
        state_dim = env.observation_space.shape[2] # Channels usually 3
        use_cnn = True
    else:
        print(f"Detected Vector Environment: {env_name}")
        state_dim = env.observation_space.shape[0]
    
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize Agent
    if model_type == "SAC":
        agent = SACAgent(state_dim, action_dim, env.action_space, config, use_cnn)
    elif model_type == "TD3":
        agent = TD3Agent(state_dim, action_dim, max_action, config, use_cnn)
    elif model_type == "PPO":
        # Pass action_space for correct scaling (Critical Fix)
        agent = PPOAgent(state_dim, action_dim, config, use_cnn=use_cnn, action_space=env.action_space)

    print(f"Training {model_type} on {env_name} for {timesteps} steps...")
    
    state, _ = env.reset()
    episode_reward = 0
    episode_timesteps = 0
    
    for t in range(int(config['total_timesteps'])):
        episode_timesteps += 1
        
        # Action Selection
        if model_type == "TD3":
            action = agent.select_action(state)
            noise = np.random.normal(0, 0.1, size=action_dim)
            action = (action + noise).clip(-max_action, max_action)
        else:
            action = agent.select_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Storage & Updates
        if model_type == "PPO":
            agent.store_transition(state, action, reward, done)
        else:
            agent.store_transition(state, action, reward, next_state, done)
            if t > config['batch_size']:
                loss = agent.update(config['batch_size'])
                if t % 100 == 0: wandb.log({"train/loss": loss})

        state = next_state
        episode_reward += reward

        if done:
            if model_type == "PPO":
                loss = agent.update()
                wandb.log({"train/loss": loss})
            
            wandb.log({"train/reward": episode_reward, "train/episode_len": episode_timesteps})
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0

        # Evaluation & Video Recording Loop
        if (t + 1) % config['eval_freq'] == 0:
            evaluate_policy(agent, env_name, model_type, max_action, t+1)
            agent.save(f"saved_models/{model_type}_{env_name}_{t+1}.pth")

    # Final Save
    agent.save(f"saved_models/{model_type}_{env_name}_final.pth")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["SAC", "TD3", "PPO"])
    parser.add_argument("--env", type=str, required=True, choices=["LunarLander-v3", "CarRacing-v3"])
    parser.add_argument("--timesteps", type=int, default=None, help="Override training timesteps (optional)")
    args = parser.parse_args()
    
    train(args.model, args.env, timesteps=args.timesteps)
