import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from PPO2 import PPO, ActorCritic
import torch
import matplotlib.pyplot as plt
import numpy as np
import random


# -------------------------
# Environment Setup
# -------------------------

class Environments():
    def __init__(self, nb_actor):
        self.envs = [self.get_env() for _ in range(nb_actor)]
        self.observations = [None for _ in range(nb_actor)]
        self.current_life = [None for _ in range(nb_actor)]
        self.done = [False for _ in range(nb_actor)]
        self.total_rewards = [0 for _ in range(nb_actor)]
        self.nb_actor = nb_actor

        for env_id in range(nb_actor):
            self.reset_env(env_id)

    def len(self):
        return self.nb_actor

    def reset_env(self, env_id):
        """Reset environment with proper initialization sequence for Atari games"""
        self.total_rewards[env_id] = 0
        obs, info = self.envs[env_id].reset()
        self.observations[env_id] = obs
        self.current_life[env_id] = info['lives']
        self.done[env_id] = False

        # Fire to start game
        self.observations[env_id], reward, terminated, truncated, info = self.envs[env_id].step(1)  # Fire

        # Random no-op steps for better exploration
        for _ in range(random.randint(1, 30)):  # No-ops
            if terminated or truncated:
                obs, info = self.envs[env_id].reset()
                self.observations[env_id] = obs
                # Fire again after reset
                self.observations[env_id], reward, terminated, truncated, info = self.envs[env_id].step(1)
                break
            self.observations[env_id], reward, terminated, truncated, info = self.envs[env_id].step(0)  # No-op

        self.current_life[env_id] = info['lives']

    def step(self, env_id, action):
        """Take a step in the environment"""
        next_obs, reward, terminated, truncated, info = self.envs[env_id].step(action)
        done = terminated or truncated

        # Use life loss as a terminal signal
        dead = info.get('lives', 0) < self.current_life[env_id]
        self.done[env_id] = done
        self.total_rewards[env_id] += reward
        self.current_life[env_id] = info['lives']
        self.observations[env_id] = next_obs
        return next_obs, reward, dead, done, info

    def get_env(self):
        """Create a properly configured Atari environment"""
        gym.register_envs(ale_py)
        env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0.0)
        env = AtariPreprocessing(
            env,
            grayscale_obs=True,
            scale_obs=False,  # Don't scale within the wrapper
            terminal_on_life_loss=True,  # Important for Breakout
            screen_size=84
        )
        env = FrameStackObservation(env, stack_size=4)
        return env

    def get_env_human(self):
        """Create an environment for human viewing"""
        gym.register_envs(ale_py)
        env = gym.make("ALE/Breakout-v5", frameskip=1, render_mode="human", repeat_action_probability=0.0)
        env = AtariPreprocessing(
            env,
            grayscale_obs=True,
            scale_obs=False,
            terminal_on_life_loss=True,
            screen_size=84
        )
        env = FrameStackObservation(env, stack_size=4)
        return env


def evaluate_model(env, model, device='cuda', num_episodes=10, render=False):
    """
    Runs the given model in 'env' for 'num_episodes' episodes and returns the average reward.
    Renders the environment if render=True.
    """
    model.eval()  # set model to evaluation mode
    rewards = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        # Start the game with FIRE
        obs, reward, terminated, truncated, info = env.step(1)
        episode_reward += reward
        done = terminated or truncated

        while not done:
            # Normalize the observation (divide by 255.0)
            obs_tensor = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to(device) / 255.0

            with torch.no_grad():
                # Get action deterministically for evaluation
                action, _ = model.get_action(obs_tensor, deterministic=True)
                action = action.item()


            # Step the environment using the chosen action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            if render:
                env.render()

        rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")

    env.close()
    avg_reward = np.mean(rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
    return avg_reward


# -------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Number of parallel environments
    nb_actor = 16

    # Create environments
    envs = Environments(nb_actor)

    # Create model
    model = ActorCritic(envs.envs[0].action_space.n).to(device)

    # Checkpoint path (set to None for fresh training)
    checkpoint_path = "checkpoint_9000.pth"  # "checkpoint_30000.pth"  # Set to None for fresh training

    # Train the model 6300
    # PPO(envs, model, device=device, checkpoint_path=checkpoint_path,
    #     T=256,
    #     K=4,  # 4 epochs per update
    #     batch_size=128,
    #     learning_rate=2.5e-4,
    #     gamma=0.99,
    #     vf_coeff_c1=0.5,  # Value loss coefficient
    #     ent_coef_c2=0.01,  # Entropy coefficient
    #     nb_iterations=40_000)

    # To evaluate a trained model, uncomment these lines:
    # checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # model.load_state_dict(checkpoint["actorcritic_state"])
    # model.to(device)
    # # Test environment with rendering
    # test_env = envs.get_env_human()
    # average_reward = evaluate_model(test_env, model, device=device, num_episodes=10, render=True)
    # print("Evaluation done. Average reward:", average_reward)