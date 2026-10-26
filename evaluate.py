"""Evaluate a saved DQN checkpoint on fresh episodes."""
import argparse, json, os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(state_dim,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU(), nn.Linear(128,action_dim))
    def forward(self, x): return self.network(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="results/dqn_cartpole.pth")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2000)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    model = DQN(ckpt["state_dim"], ckpt["action_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    env = gym.make("CartPole-v1"); scores = []
    for ep in range(args.episodes):
        state, _ = env.reset(seed=args.seed + ep); total = 0
        while True:
            with torch.no_grad():
                x = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(model(x).argmax(1).item())
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated: break
        scores.append(total)
    env.close()
    result = {"environment":"CartPole-v1", "evaluation_episodes":len(scores),
        "average_reward":float(np.mean(scores)), "standard_deviation":float(np.std(scores)),
        "best_reward":float(max(scores)), "rewards":scores}
    print(json.dumps(result, indent=2))
    os.makedirs("results", exist_ok=True)
    with open("results/evaluation_results.json","w") as f: json.dump(result, f, indent=2)

if __name__ == "__main__": main()
