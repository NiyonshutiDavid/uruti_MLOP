# **Pitch Coach – Reinforcement Learning for Presentation Skills**

An AI-powered reinforcement learning system designed to help founders and presenters improve their pitch delivery. Pitch Coach simulates a dynamic presentation environment where an RL agent learns to optimize confidence, engagement, clarity, and pacing—ultimately providing meaningful feedback for better pitch performance.

---

## 📋 **Table of Contents**

* [Project Overview](#-project-overview)
* [Environment Description](#-environment-description)
* [System Architecture](#-system-architecture)
* [Installation & Setup](#️-installation--setup)
* [Project Structure](#-project-structure)
* [Usage](#-usage)
* [Training Results](#-training-results)
* [Report Structure](#-report-structure)
* [Demo & Visualization](#-demo--visualization)
* [Troubleshooting](#-troubleshooting)
* [License](#-license)
* [Acknowledgments](#-acknowledgments)

---

## 🎯 **Project Overview**

Pitch Coach addresses a common challenge: **many founders struggle to deliver compelling pitches**, often lacking objective feedback mechanisms to improve their delivery.

### **Solution**

A simulated pitch environment where reinforcement learning agents learn effective presentation strategies through real-time feedback on:

* Confidence
* Audience engagement
* Message clarity
* Pacing & slide progression

The system uses **Stable-Baselines3** and implements **DQN, PPO (REINFORCE), and A2C** to determine the best-performing RL method in this interactive skill-learning context.

---

## 🎮 **Environment Description**

### **Agent**

The agent simulates a presenter delivering a pitch and can:

* Adjust presentation style and energy
* Manage slide transitions
* Use engagement techniques (gestures, eye contact, storytelling)
* Adapt based on audience feedback
* Optimize clarity, engagement, and confidence

---

### **Action Space (Discrete – 6 Actions)**

| Action | Description                 |
| ------ | --------------------------- |
| 0      | Maintain presentation style |
| 1      | Increase energy             |
| 2      | Use gestures                |
| 3      | Make eye contact            |
| 4      | Next slide                  |
| 5      | Add storytelling            |

---

### **Observation Space (6-D Continuous Vector)**

`[confidence, engagement, clarity, pace, slide_progress, time_remaining]`

* **confidence** (0–1)
* **engagement** (0–1)
* **clarity** (0–1)
* **pace** (0–2)
* **slide_progress** (0–1)
* **time_remaining** (0–1)

---

### **Reward Function**

```
R(s,a) = R_action(a) + 0.1 * (0.3*confidence + 0.4*engagement + 0.3*clarity)
```

**Bonuses**

* +10 × slide_progress
* +10 for completing the full presentation

**Penalties**

* Natural decay: -0.02 to -0.03 per step
* Invalid slide advance: -0.2

---

## 🏗️ **System Architecture**

### **Deep Q-Network (DQN)**

* Input: 6 features
* Hidden layers: 128 → 64
* Outputs: Q-values for 6 actions
* Experience Replay (10,000)
* Target Network
* Epsilon-Greedy (0.1 → 0.01)
* Huber Loss

---

### **PPO (REINFORCE)**

* Actor: 64 → 32 → softmax(6)
* Critic: 64 → 32 → state value
* Features:

  * Clipped objective
  * GAE
  * Entropy regularization (β=0.01)

---

### **A2C**

* Shared feature extractor (128 units)
* Actor & Critic heads
* 5 parallel workers
* N-step returns

---

## ⚙️ **Installation & Setup**

### **Prerequisites**

* Python 3.8+
* `pip`

---

### **Quick Install**

```bash
# Clone repository
git clone https://github.com/your-username/pitch-coach-rl.git
cd pitch-coach-rl

# Create environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Test installation
python test_core_functionality.py
```

---

### **Manual Dependency Installation**

```bash
pip install torch==2.1.0
pip install gymnasium==0.29.1
pip install stable-baselines3==2.0.0

# Visualization
pip install pygame opencv-python matplotlib

# Utilities
pip install numpy pandas tensorboard

# (Optional) Audio
brew install portaudio
pip install pyaudio
```

---

## 📂 **Project Structure**

```
pitch-coach-rl/
├── environment/
│   ├── pitch_env.py
│   └── simple_pitch_env.py
├── utils/
│   ├── visualization.py
│   └── audio_processor.py
├── train.py
├── plat.py
├── run_play.py
├── training_config.json
└── requirements.txt
```

---

## 🚀 **Usage**

### **Training**

```bash
# Train DQN
python train.py --algorithm dqn --config training_config.json

# Train all algorithms
python train.py --algorithm all --config training_config.json

# Custom training
python train.py --algorithm reinforce --total_timesteps 50000
```

---

### **Evaluate Models**

```bash
python plat.py --model experiments/dqn_model.zip --algorithm dqn --save-video

python run_play.py   # Interactive selector

python plat.py --model experiments/reinforce_model.zip --record-audio --save-video
```

---

### **Monitor Training**

```bash
tensorboard --logdir experiments/
```

---

## 📊 **Training Results**

### **DQN Hyperparameters (Sample of 10 Runs)**

| LR     | Gamma | Buffer | Batch | Exploration | Mean Reward |
| ------ | ----- | ------ | ----- | ----------- | ----------- |
| 0.0001 | 0.99  | 10000  | 32    | 0.1→0.01    | 8.45        |
| …      | …     | …      | …     | …           | …           |

*(Full table included in your report section)*

---

### **REINFORCE (PPO)** and **A2C** results also included with full tables.

---

## 📈 **Results Discussion**

### **Summary**

* **REINFORCE (PPO)** achieved the highest and most stable performance.
* **DQN** learned faster initially but showed high variance.
* **A2C** performed moderately with smooth improvements.
* REINFORCE generalized best to unseen scenarios.

### **Convergence**

* **REINFORCE**: ~150 episodes
* **DQN**: ~200 episodes
* **A2C**: ~180 episodes

---

## 📋 **Report Structure (For Students)**

Includes:

* Project Overview
* Environment Description
* System Analysis & Design
* Implementation (hyperparameter tables)
* Results Discussion
* Conclusion & Future Work

*(All sections rewritten clearly in the README for reference.)*

---

## 🎥 **Demo & Visualization**

### **Screenshots**

* Training curves
* Agent demo
* Environment visualization

(Images stored in `/docs/images/`)

### **Video Demo**

A 3-minute video showing:

* Agent interacting with the environment
* Real-time metric changes
* Performance reporting

---

## 🛠️ **Troubleshooting**

```bash
# Pygame issues
python test_pygame.py

# Audio issues
brew install portaudio
pip install pyaudio

# Model loading problems
python run_play.py
```

**Performance Tips**

* Use GPU for training
* Reduce environment complexity for quick debugging
* Lower total_timesteps for faster experiments

---

## 📄 **License**

This project is licensed under the **MIT License**.

---

## 🙏 **Acknowledgments**

* Stable-Baselines3 developers
* Gymnasium contributors
* Pygame community

