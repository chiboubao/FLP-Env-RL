# RL-FLP: Custom Reinforcement Learning Environment

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Python](https://img.shields.io/badge/python-3.8%2B-green)
![Framework](https://img.shields.io/badge/framework-OpenAI%20Gym-orange)
![GCN](https://img.shields.io/badge/Graph-GCN-brightgreen)

## English | [中文](README_CN.md)

**RL-FLP** is a "smart" reinforcement learning environment specifically optimized for **Facilities Layout Problems (FLP)**. Unlike standard flat-vector environments, it utilizes a graph-based architecture to capture spatial and relational dependencies in complex industrial systems.

The implementation is **generic and extensible**, supporting:
* **Static FLPs**: Optimization for a single period where material flow and demand are constant and known.
* **Dynamic FLPs**: Optimization across multiple periods where material flow changes, requiring layout reconfigurations to maintain efficiency.
* **Stochastic FLPs**: Optimization where material flows are uncertain and modelled with a probability distribution.

The environment’s goal is to build a robust, stable, and mature framework for training agents to minimize material handling costs and maximize layout efficiency for every type of FLPs

---

## Features

* **Gym-Compatible API**: Seamlessly integrates with standard RL libraries like Stable Baselines3, Ray Rllib, or custom TensorFlow/Keras loops.
* **Graph-Based Representation**: Models machines as nodes, allowing for complex relational reasoning via an adjacency matrix.
* **Integrated GCN**: A two-layer Graph Convolutional Network (GCN) captures spatial features up to two "hops" away, providing local and global context.
* **Shape-Agnostic Design**: Uses Global Average Pooling to handle varying layout sizes (e.g., 36 workstations on a 121-location grid) without breaking the neural network architecture.
* **Modular & Agnostic**: Compatible with a wide variety of algorithms including **DQN, PPO, A2C, and Rainbow DQN**.
* **Hybrid State Encoding**: A multimodal state vector combining graph-level features, workstation coordinates, and transportation demand flows.

---

## Environment Design

### Machines as Graph Nodes
Each workstation is represented as a node in a graph. Nodes are associated with **4 positional features**:
* **Pickup location** $(x, y)$
* **Delivery location** $(x, y)$



### Graph Representation
The layout is modeled using **NetworkX**:
1. **Numbered Nodes**: Represent the physical sites to host workstations.
2. **P and D Nodes**: Represent the physical workstations where P is where product are collected and D is where products are delivered
3. **Edges**: Represent relationships, interactions, or material flow between machines.
4. **Self-loops**: Automatically added to the adjacency matrix to preserve node identity during graph convolutions.

---

## States Encoding

The environment uses a specialized spatial processing pipeline to encode the layout before the agent selects an action:

1. **Preprocessing**: Extracts the adjacency matrix $A$ and node features $X$.
2. **Message Passing (2-Hops)**: 
   * **Layer 1**: Aggregates neighbor information ($A \cdot X \cdot W_1$) with 16 hidden units + ReLU.
   * **Layer 2**: Aggregates 2nd-hop neighbor information ($A \cdot H_1 \cdot W_2$) with 8 hidden units + ReLU.
3. **Global Pooling**: Node embeddings are condensed into a fixed-size graph-level feature vector via **Global Average Pooling**, ensuring compatibility across different graph sizes.
4. **P and D coordinates**: are normalized
5. **Transportation demands**: are normalized in the case of Stochastic and Dynamic FLPs.

---

## Quick Start / Example Usage

```python
from Agent_Environment import Agent_Environment

# Initialize the environment
env = Agent_Environment(
    number_of_machines=36,
    number_of_steps=10,
    reward_type='scaled_cost',
    render=False
)

# Reset and get initial observation
state, init_cost = env.reset()

# Sample a random action
action = env.action_space.sample()
next_state, reward, done = env.step(action)

print(f"Current State Shape: {next_state.shape}")
print(f"Reward Received: {reward}")

```

---

## Installation

```bash
# Clone the repository
git clone [https://github.com/chiboubao/FLP-Env-RL.git](https://github.com/chiboubao/FLP-Env-RL.git)

# Navigate to the folder
cd FLP-Env-RL

# Install dependencies
pip install tensorflow numpy networkx gym

```


