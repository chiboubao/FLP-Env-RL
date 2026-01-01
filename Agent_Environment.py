import sys
import gym
import numpy as np
import tensorflow as tf
import networkx as nx
from Environment import *
from Environment_FLP.state_processing import *
from tensorflow.python.keras.layers import Input, Dense, Lambda, Flatten, concatenate
from tensorflow.python.keras.models import Model

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class Agent_Environment:
    def __init__(self, number_of_machines, number_of_steps, reward_type, render):
        self.agent = Env(number_of_machines, number_of_steps, number_of_steps, reward_type,
                         [number_of_machines], render)
        self.num_machines = number_of_machines
        self.env_state = None
        self.init_cost = None
        self.positions = None
        self.current_G = None

        self.num_actions = len(self.agent.action_name_dict)
        self.action_space = gym.spaces.Discrete(self.num_actions)

        # Initialize the GCN model
        self.gcn_model = self.build_gcn_extractor()

    def build_gcn_extractor(self):
        # Adjacency matrix (N x N)
        input_adj = Input(shape=(self.num_machines, self.num_machines))

        # Node features (N x F), here F = 4
        input_features = Input(shape=(self.num_machines, 4))

        def normalize_adjacency(A):
            """
            Compute D^{-1/2} (A + I) D^{-1/2}
            """
            # Add self-loops
            A_hat = A + tf.eye(self.num_machines)

            # Degree matrix
            D = tf.reduce_sum(A_hat, axis=-1)
            D_inv_sqrt = tf.linalg.diag(tf.pow(D, -0.5))

            return tf.matmul(tf.matmul(D_inv_sqrt, A_hat), D_inv_sqrt)

        # Normalize adjacency
        A_norm = Lambda(normalize_adjacency)(input_adj)

        # -------- GCN Layer 1 --------
        x = Lambda(lambda x: tf.matmul(x[0], x[1]))([A_norm, input_features])
        x = Dense(16, activation='relu')(x)

        # -------- GCN Layer 2 --------
        x = Lambda(lambda x: tf.matmul(x[0], x[1]))([A_norm, x])
        x = Dense(8, activation='relu')(x)

        # Readout
        output = Flatten()(x)

        return Model(inputs=[input_adj, input_features], outputs=output)

    @staticmethod
    def normalize_position_nodes(vector, min_val: int, max_val: int):
        # Handle dictionary input if pos is passed as a dict of coordinates
        if isinstance(vector, dict):
            vector = np.array(list(vector.values()))
        normalized_vector = [(x - min_val) / (max_val - min_val) for x in vector]
        return np.array(normalized_vector)

    def get_state(self, G, pos, demand):

        if G.number_of_nodes() != self.num_machines:
            raise ValueError(
                f"Graph has {G.number_of_nodes()} nodes, "
                f"but num_machines = {self.num_machines}"
            )
        # 1. Prepare Adjacency
        adj = nx.to_numpy_array(G)
        adj = adj + np.eye(self.num_machines)  # Shape: (36, 36)

        # 2. Normalize and RESHAPE node features
        # We need to ensure pos_array is (36, 2)
        pos_array = self.normalize_position_nodes(pos, 0, 6)

        # FIX: Explicitly reshape to (Number of machines, 2 coordinates)
        pos_features = np.array(pos_array).reshape(self.num_machines, 4)

        demand_array = np.array(demand).flatten()
        processed_demand = self.normalize_position_nodes(demand_array, 0, 40)

        # 3. Graph Feature Extraction
        # Add the batch dimension to make it (1, 36, 36) and (1, 36, 2)
        inputs = [
            tf.convert_to_tensor(adj[np.newaxis, :], dtype=tf.float32),
            tf.convert_to_tensor(pos_features[np.newaxis, :], dtype=tf.float32)
        ]

        graph_features_tensor = self.gcn_model(inputs, training=False)
        graph_features = graph_features_tensor.numpy().flatten()

        # 4. Concatenation
        # We use pos_features.flatten() here to keep the final state vector flat for the RL agent
        combined_state = np.concatenate((graph_features, pos_features.flatten(), processed_demand))
        return combined_state

    def reset(self):
        self.env_state, self.init_cost, self.positions, demand = self.agent.reset()
        # Ensure we have an initial graph structure
        self.current_G = nx.complete_graph(self.num_machines)
        initial_state = self.get_state(self.current_G, self.positions, demand)
        return initial_state, self.init_cost

    def step(self, action):
        G, env_next_state, reward, done, next_cost, position, demand = self.agent.step(
            self.env_state, action, self.init_cost)
        G = nx.Graph()
        G.add_nodes_from(range(self.num_machines))
        # optionally: update edges or weights
        self.current_G = G
        next_state = self.get_state(G, position, demand)
        self.env_state = env_next_state
        self.init_cost = next_cost

        return next_state, reward, done

    def close(self):
        self.agent.close()


# --- Execution ---
env = Agent_Environment(number_of_machines=36, number_of_steps=10, reward_type='scaled_cost', render=True)

total_reward = 0
for _ in range(40):  # episodes
    observation, _ = env.reset()
    for _ in range(len(env.agent.action_name_dict)):  # steps
        action = env.action_space.sample()
        next_observation, reward, done = env.step(action)
        total_reward += reward

env.close()
print("Total reward:", total_reward)
