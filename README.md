# Deep RL CartPole Agent

A Deep Q-Network (DQN) agent implemented in PyTorch and trained in Gymnasium's `CartPole-v1` environment. It learns through trial and error to choose left/right actions that keep a pole balanced on a moving cart.

## Features
- DQN neural network with two 128-unit hidden layers
- Experience replay and epsilon-greedy exploration
- Target network for more stable Q-learning targets
- Training reward curve and saved checkpoint
- Separate evaluation script for fresh episodes

## Results from the development experiment
The recorded Colab run used 500 training episodes. In the final evaluation, the policy scored **500/500 in all 20 evaluation episodes** (mean 500.0, standard deviation 0.0). This is specific to the recorded run and seed set; results can vary across reruns. The training run's final-20-episode average was 317.65, so training and evaluation results are reported separately.

These results were observed in a notebook run. Re-run the included scripts to reproduce results in your own environment; no claim is made that every seed or retraining run will achieve the same score.

## Setup
Python 3.10+ recommended.

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Train
```bash
python train.py
```
Checkpoint, reward history and training plot are saved under `results/`.

## Evaluate
```bash
python evaluate.py --episodes 20
```
Evaluation metrics are printed and saved to `results/evaluation_results.json`.

## Algorithm
1. Observe cart position, cart velocity, pole angle and pole angular velocity.
2. Choose an action with epsilon-greedy exploration.
3. Store `(state, action, reward, next_state, done)` transitions in replay memory.
4. Sample random mini-batches and optimize the temporal-difference target.
5. Periodically copy policy weights to the target network.

## Tech stack
Python · PyTorch · Gymnasium · NumPy · Matplotlib

## Future improvements
- Run multiple random seeds and report mean ± standard deviation.
- Implement Double DQN or soft target updates.
- Add an evaluation video and random-action baseline.

## License
MIT. See `LICENSE`.
