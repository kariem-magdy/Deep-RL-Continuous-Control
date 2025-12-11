# 🏎️ Deep RL: Continuous Control with PPO, SAC, and TD3

This project implements and compares three state-of-the-art Deep Reinforcement Learning algorithms—**PPO**, **SAC**, and **TD3**—to solve continuous control tasks in Gymnasium environments, specifically focusing on the pixel-based **`CarRacing-v3`** and vector-based **`LunarLander-v3`**.

The implementation features custom neural network architectures, robust handling of image observations, and integration with **Weights & Biases (WandB)** for experiment tracking.

## 🚀 Features

  * **Algorithms**:
      * **PPO** (Proximal Policy Optimization): On-policy, stochastic.
      * **SAC** (Soft Actor-Critic): Off-policy, entropy-regularized, stochastic.
      * **TD3** (Twin Delayed DDPG): Off-policy, deterministic, reduces overestimation bias.
  * **Environments**:
      * `CarRacing-v3`: Learn to drive from raw pixels (96x96x3).
      * `LunarLander-v3`: Continuous lander control from vector states.
  * **Robust Architecture**:
      * Custom **CNN Feature Extractor** for image-based tasks.
      * **Action Repeat Wrapper** for smoother control and faster training in CarRacing.
      * **NaN Safety Patches**: Custom handling for Gymnasium v1.0+ action space edge cases.
  * **Experiment Tracking**: Automatic logging of rewards, losses, and evaluation videos to WandB.

## 📂 Project Structure

```bash
.
├── agents/                 # Agent implementations
│   ├── PPO.py             # Proximal Policy Optimization
│   ├── SAC.py             # Soft Actor-Critic
│   └── TD3.py             # Twin Delayed DDPG
├── utils/
│   ├── NNArch.py          # Neural Network definitions (Actor, Critic, CNN)
│   ├── ReplayBuffer.py    # Standard buffer for vector envs
│   └── ImageReplayBuffer.py # Optimized buffer for image envs
├── train.py               # Main training script
├── requirements.txt       # Python dependencies
└── upload_to_hf.py        # Utility to push models to Hugging Face
```

## 🛠️ Installation

### 1\. System Dependencies

This project requires **Box2D** and **SWIG** for the physics engine, and `ffmpeg` for video recording.

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y xvfb ffmpeg swig
```

### 2\. Python Dependencies

Install the required Python packages.

```bash
pip install -r requirements.txt
# OR manually:
pip install "gymnasium[box2d]>=1.0.0" torch numpy wandb moviepy imageio
```

## 💻 Usage

To train an agent, run `train.py` with the desired model and environment.

### Basic Command

```bash
python train.py --model SAC --env CarRacing-v3
```

### Arguments

| Argument | Description | Choices / Default |
| :--- | :--- | :--- |
| `--model` | The RL algorithm to use. | `SAC`, `TD3`, `PPO` |
| `--env` | The Gymnasium environment. | `CarRacing-v3`, `LunarLander-v3` |
| `--timesteps` | Total training steps. | Default: `100,000` (Vector) / `200,000` (Image) |
| `--wandb_project`| Name of the W\&B project. | Default: `RL_Assignment4` |

### Examples

**Train PPO on LunarLander:**

```bash
python train.py --model PPO --env LunarLander-v3 --timesteps 100000
```

**Train TD3 on CarRacing (Long Run):**

```bash
python train.py --model TD3 --env CarRacing-v3 --timesteps 200000 --wandb_project My_Car_Racing
```

## 🧠 Architecture Details

### 1\. Neural Networks (`utils/NNArch.py`)

The project automatically detects if the environment is image-based (3D observation) or vector-based (1D observation).

  * **Visual Encoder:** A 3-layer Convolutional Neural Network (CNN) extracts a 256-dim feature vector from 96x96 images.
  * **Policy Networks:**
      * *GaussianPolicy (SAC/PPO):* Outputs mean and log-std for stochastic sampling. Includes Tanh squashing.
      * *DeterministicPolicy (TD3):* Outputs direct action values with Tanh activation scaled to `max_action`.
  * **Critic Networks:**
      * *DoubleQNetwork (SAC/TD3):* Twin Q-networks to mitigate overestimation bias.
      * *ValueNetwork (PPO):* Estimates state value V(s).

### 2\. Critical Fixes

This codebase includes patches for common issues found in newer Gymnasium versions:

  * **NaN Action Handling:** The architecture safely sanitizes `action_space` bounds to prevent crashes when `env.action_space.high` contains NaNs.
  * **Tensor Shapes:** Fixed `view()` vs `reshape()` mismatches in the CNN forward pass.
  * **Action Repeat:** Implements an `ActionRepeatWrapper` (repeats action 3 times) to stabilize steering behavior in `CarRacing`.

## 📊 Results & Logging

Training progress is logged to **Weights & Biases**.

  * **Metrics:** Episode Reward, Episode Length, Actor/Critic Loss.
  * **Videos:** Every `eval_freq` steps, the agent is evaluated, and a video of the episode is uploaded to WandB.

## 🤝 Contributing

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/NewAlgo`).
3.  Commit your changes.
4.  Push to the branch and open a Pull Request.

## 📜 License

This project is open-source and available under the MIT License.