import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# Environment parameters
GYM_ID = "LunarLander-v3"
NUM_ENVS = 16
CAPTURE_VIDEO = False

# Training parameters
TOTAL_TIMESTEPS = 10000000
LEARNING_RATE = 2.5e-4
SEED = 1
TORCH_DETERMINISTIC = True
CUDA = True

# PPO specific parameters
NUM_STEPS = 128
ANNEAL_LR = True
GAE = True
GAMMA = 0.99
GAE_LAMBDA = 0.95
NUM_MINIBATCHES = 4
UPDATE_EPOCHS = 4
NORM_ADV = True
CLIP_COEF = 0.2  # Increased for Lunar Lander (0.1->0.2)
CLIP_VLOSS = True
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
TARGET_KL = None

# Derived parameters
BATCH_SIZE = NUM_ENVS * NUM_STEPS
MINIBATCH_SIZE = BATCH_SIZE // NUM_MINIBATCHES

# Tracking parameters
TRACK = True
WANDB_PROJECT_NAME = "ppo-lunar-lander"
WANDB_ENTITY = None
EXP_NAME = "ppo-lunar-lander"

# Checkpointing parameters
CHECKPOINT_FREQUENCY = 10  # Save a checkpoint every 10 updates
CHECKPOINT_DIR = "checkpoints"
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# Generate run name
RUN_NAME = f"LunarLander-v2__{EXP_NAME}__{SEED}__{int(time.time())}"


def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id, render_mode="rgb_array" if capture_video and idx == 0 else None)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
                episode_trigger=lambda x: x % 100 == 0
            )  # Record every 100 episodes
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        # Modified for Lunar Lander - uses a simpler MLP architecture
        # instead of CNNs since the input is a state vector, not an image
        self.network = nn.Sequential(
            layer_init(nn.Linear(envs.single_observation_space.shape[0], 64)),
            nn.Tanh(),  # Tanh often works better for control tasks
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.actor = layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(64, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


if __name__ == "__main__":
    # Set up logging
    writer = SummaryWriter(f"runs/{RUN_NAME}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" +
        "\n".join([f"|GYM_ID|{GYM_ID}|",
                   f"|NUM_ENVS|{NUM_ENVS}|",
                   f"|TOTAL_TIMESTEPS|{TOTAL_TIMESTEPS}|",
                   f"|LEARNING_RATE|{LEARNING_RATE}|",
                   f"|SEED|{SEED}|",
                   f"|NUM_STEPS|{NUM_STEPS}|",
                   f"|ANNEAL_LR|{ANNEAL_LR}|",
                   f"|GAE|{GAE}|",
                   f"|GAMMA|{GAMMA}|",
                   f"|GAE_LAMBDA|{GAE_LAMBDA}|",
                   f"|NUM_MINIBATCHES|{NUM_MINIBATCHES}|",
                   f"|UPDATE_EPOCHS|{UPDATE_EPOCHS}|",
                   f"|NORM_ADV|{NORM_ADV}|",
                   f"|CLIP_COEF|{CLIP_COEF}|",
                   f"|CLIP_VLOSS|{CLIP_VLOSS}|",
                   f"|ENT_COEF|{ENT_COEF}|",
                   f"|VF_COEF|{VF_COEF}|",
                   f"|MAX_GRAD_NORM|{MAX_GRAD_NORM}|",
                   f"|TARGET_KL|{TARGET_KL}|",
                   f"|BATCH_SIZE|{BATCH_SIZE}|",
                   f"|MINIBATCH_SIZE|{MINIBATCH_SIZE}|"]),
    )

    # Set up wandb if tracking is enabled
    if TRACK:
        import wandb

        # Ensure user is logged in
        wandb.login()

        wandb.init(
            project=WANDB_PROJECT_NAME,
            entity=WANDB_ENTITY,
            dir=os.path.abspath("./runs"),
            sync_tensorboard=False,  # Disable TB sync since we're using direct logging
            config={
                "GYM_ID": GYM_ID,
                "NUM_ENVS": NUM_ENVS,
                "TOTAL_TIMESTEPS": TOTAL_TIMESTEPS,
                "LEARNING_RATE": LEARNING_RATE,
                "SEED": SEED,
                "NUM_STEPS": NUM_STEPS,
                "ANNEAL_LR": ANNEAL_LR,
                "GAE": GAE,
                "GAMMA": GAMMA,
                "GAE_LAMBDA": GAE_LAMBDA,
                "NUM_MINIBATCHES": NUM_MINIBATCHES,
                "UPDATE_EPOCHS": UPDATE_EPOCHS,
                "NORM_ADV": NORM_ADV,
                "CLIP_COEF": CLIP_COEF,
                "CLIP_VLOSS": CLIP_VLOSS,
                "ENT_COEF": ENT_COEF,
                "VF_COEF": VF_COEF,
                "MAX_GRAD_NORM": MAX_GRAD_NORM,
                "TARGET_KL": TARGET_KL,
                "BATCH_SIZE": BATCH_SIZE,
                "MINIBATCH_SIZE": MINIBATCH_SIZE
            },
            name=RUN_NAME,
            monitor_gym=True,
            save_code=True,
            resume="allow"  # Allow resuming if wandb detects incomplete runs
        )

    # Seeding
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = TORCH_DETERMINISTIC

    device = torch.device("cuda" if torch.cuda.is_available() and CUDA else "cpu")

    # Environment setup
    episode_rewards = [0] * NUM_ENVS
    episode_lengths = [0] * NUM_ENVS

    # Create directory for checkpoints
    os.makedirs(f"{CHECKPOINT_DIR}/{RUN_NAME}", exist_ok=True)

    envs = gym.vector.SyncVectorEnv(
        [make_env(GYM_ID, SEED + i, i, CAPTURE_VIDEO, RUN_NAME) for i in range(NUM_ENVS)]
    )
    print(f"Number of environments: {len(envs.envs)}")
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    # Check for existing checkpoint
    checkpoint_files = []
    if os.path.exists(f"{CHECKPOINT_DIR}/{RUN_NAME}"):
        checkpoint_files = [f for f in os.listdir(f"{CHECKPOINT_DIR}/{RUN_NAME}") if f.endswith(".pt")]

    start_update = 1
    global_step = 0

    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        checkpoint_path = os.path.join(f"{CHECKPOINT_DIR}/{RUN_NAME}", latest_checkpoint)
        print(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        agent.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = checkpoint["global_step"]
        start_update = checkpoint["update"] + 1

        # Restore random states
        random.setstate(checkpoint["random_state"]["python"])
        np.random.set_state(checkpoint["random_state"]["numpy"])
        torch.set_rng_state(checkpoint["random_state"]["torch"])
        if torch.cuda.is_available() and checkpoint["random_state"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["random_state"]["cuda"])

        print(f"Resuming training from update {start_update}, global step {global_step}")

    # Storage setup
    obs = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    rewards = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    dones = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    values = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)

    # Start the game
    start_time = time.time() if global_step == 0 else time.time() - checkpoint.get("elapsed_time", 0)
    reset_result = envs.reset()
    next_obs = torch.Tensor(reset_result[0] if isinstance(reset_result, tuple) else reset_result).to(device)
    next_done = torch.zeros(NUM_ENVS).to(device)
    num_updates = TOTAL_TIMESTEPS // BATCH_SIZE

    for update in range(start_update, num_updates + 1):
        # Annealing the learning rate if instructed to do so
        if ANNEAL_LR:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * LEARNING_RATE
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, NUM_STEPS):
            global_step += 1 * NUM_ENVS
            obs[step] = next_obs
            dones[step] = next_done

            # Action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute the game and log data
            step_result = envs.step(action.cpu().numpy())
            if len(step_result) == 5:  # New Gymnasium API
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated | truncated  # Combine both signals into a single done flag
            else:  # Old Gym API
                next_obs, reward, done, info = step_result

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            # Episode tracking and logging
            for i, (rew, d) in enumerate(zip(reward, done)):
                episode_rewards[i] += rew
                episode_lengths[i] += 1
                if d:
                    # Log episode metrics to wandb
                    if TRACK:
                        wandb.log({
                            "episodic_return": episode_rewards[i],
                            "episodic_length": episode_lengths[i]
                        }, step=global_step)

                    # Also log to TensorBoard
                    writer.add_scalar("charts/episodic_return", episode_rewards[i], global_step)
                    writer.add_scalar("charts/episodic_length", episode_lengths[i], global_step)

                    # Reset episode tracking
                    episode_rewards[i] = 0
                    episode_lengths[i] = 0

        # Bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if GAE:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(NUM_STEPS)):
                    if t == NUM_STEPS - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(NUM_STEPS)):
                    if t == NUM_STEPS - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + GAMMA * nextnonterminal * next_return
                advantages = returns - values

        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(BATCH_SIZE)
        clipfracs = []
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # Calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > CLIP_COEF).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if NORM_ADV:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if CLIP_VLOSS:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -CLIP_COEF,
                        CLIP_COEF,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VF_COEF

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

            if TARGET_KL is not None:
                if approx_kl > TARGET_KL:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        steps_per_sec = int(global_step / (time.time() - start_time))
        print(f"Update {update}/{num_updates}, SPS: {steps_per_sec}")
        writer.add_scalar("charts/SPS", steps_per_sec, global_step)

        # Log metrics to wandb at the end of each update
        if TRACK:
            wandb.log({
                "learning_rate": optimizer.param_groups[0]["lr"],
                "value_loss": v_loss.item(),
                "policy_loss": pg_loss.item(),
                "entropy": entropy_loss.item(),
                "approx_kl": approx_kl.item(),
                "clipfrac": np.mean(clipfracs),
                "explained_variance": explained_var,
                "steps_per_second": steps_per_sec
            }, step=global_step)

        # Save checkpoints periodically
        if update % CHECKPOINT_FREQUENCY == 0:
            checkpoint_path = os.path.join(f"{CHECKPOINT_DIR}/{RUN_NAME}", f"checkpoint_{update}.pt")
            torch.save({
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "update": update,
                "elapsed_time": time.time() - start_time,
                "random_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                }
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    envs.close()
    writer.close()
    if TRACK:
        wandb.finish()