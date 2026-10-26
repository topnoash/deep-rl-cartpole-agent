"""Train a DQN agent on Gymnasium CartPole-v1."""
import os, random
from collections import deque
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

EPISODES, CAPACITY, BATCH_SIZE = 500, 50000, 64
GAMMA, LR = 0.99, 5e-4
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 0.992
TARGET_UPDATE, SEED = 500, 42

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim))
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity): self.memory = deque(maxlen=capacity)
    def push(self, *experience): self.memory.append(experience)
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.asarray(s, np.float32), np.asarray(a, np.int64),
                np.asarray(r, np.float32), np.asarray(ns, np.float32),
                np.asarray(d, np.float32))
    def __len__(self): return len(self.memory)

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1")
    sd, ad = env.observation_space.shape[0], env.action_space.n
    policy = DQN(sd, ad).to(device); target = DQN(sd, ad).to(device)
    target.load_state_dict(policy.state_dict()); target.eval()
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    memory = ReplayBuffer(CAPACITY)
    epsilon, steps, rewards = EPS_START, 0, []

    for ep in range(EPISODES):
        state, _ = env.reset(seed=SEED + ep)
        total = 0
        while True:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    x = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = int(policy(x).argmax(1).item())
            ns, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            memory.push(state, action, reward, ns, done)
            if len(memory) >= BATCH_SIZE:
                s, a, r, next_s, d = memory.sample(BATCH_SIZE)
                s = torch.as_tensor(s, device=device)
                a = torch.as_tensor(a, device=device).unsqueeze(1)
                r = torch.as_tensor(r, device=device)
                next_s = torch.as_tensor(next_s, device=device)
                d = torch.as_tensor(d, device=device)
                q = policy(s).gather(1, a).squeeze(1)
                with torch.no_grad():
                    target_q = r + GAMMA * target(next_s).max(1).values * (1 - d)
                loss = nn.functional.smooth_l1_loss(q, target_q)
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                optimizer.step()
            state, total = ns, total + reward
            steps += 1
            if steps % TARGET_UPDATE == 0: target.load_state_dict(policy.state_dict())
            if done: break
        rewards.append(total); epsilon = max(EPS_END, epsilon * EPS_DECAY)
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1:3d}/{EPISODES} | Avg10: {np.mean(rewards[-10:]):6.1f} | Epsilon: {epsilon:.3f}")
    env.close()
    os.makedirs("results", exist_ok=True)
    torch.save({"model_state_dict": policy.state_dict(), "state_dim": sd,
                "action_dim": ad}, "results/dqn_cartpole.pth")
    np.save("results/training_rewards.npy", np.asarray(rewards))
    ma = np.convolve(rewards, np.ones(20)/20, mode="valid")
    plt.figure(figsize=(12,5))
    plt.plot(rewards, alpha=.25, label="Episode reward")
    plt.plot(range(19, len(rewards)), ma, linewidth=2, label="20-episode moving average")
    plt.axhline(500, linestyle="--", alpha=.6, label="Maximum reward")
    plt.xlabel("Episode"); plt.ylabel("Total reward")
    plt.title("DQN Training Performance — CartPole-v1")
    plt.legend(); plt.grid(True, alpha=.3); plt.tight_layout()
    plt.savefig("results/training_curve.png", dpi=150); plt.close()
    print("Saved checkpoint and plot in results/")

if __name__ == "__main__": main()
