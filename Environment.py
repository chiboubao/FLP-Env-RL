import random
import time
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from Environment_FLP.plotting import plotting_interactive
from Environment_FLP.Actions import action
from Environment_FLP.Configuration_cost import cost
from Environment_FLP.graph_corrector import corrector


def convert_keys_to_integers(d):
    return {int(key): value for key, value in d.items()}


class Env:
    def __init__(self, number_of_machines, trial_length, trial_length_max, reward, transportation_demands, render):
        self.number_of_machines = number_of_machines
        self.trial_length = trial_length
        self.trial_length_max = trial_length_max
        self.reward = reward
        self.transportation_demands = transportation_demands

        self.reset_counter = 0
        self.current_episode = 0
        self.state_visit_counter = {}

        self.random_generator = np.random.default_rng(int(time.time()))
        self.machine_shape = ['U_shape'] * self.number_of_machines

        self.G = self.load_graph()
        self.H = self.G.copy()
        self.pos = {n: d['pos'] for n, d in self.G.nodes(data=True)}

        self.sites = self.load_sites()
        self.nodes_keys = self.generate_node_keys()
        self.action_name_dict = self.generate_action_dict()

        self.observation_space = self.number_of_machines * 4 + self.number_of_machines ** 2
        self._action_space = len(self.action_name_dict)

        self.transport = None
        self.demand = None
        self.df = None

        self.render = render

    def load_graph(self):
        graph_path = f"Initial_layout/graph{self.number_of_machines}M.gml"
        return nx.read_gml(graph_path)

    def load_sites(self):
        sites_path = f"sites_data/sites{self.number_of_machines}machines.json"
        with open(sites_path, 'r') as f:
            sites = json.load(f)
        return convert_keys_to_integers(sites)

    def generate_node_keys(self):
        return [f'{label}{i}' for i in range(1, self.number_of_machines + 1) for label in ('P', 'D')]

    def generate_action_dict(self):
        action_dict = {}
        for i in range(1, self.number_of_machines + 1):
            action_dict[len(action_dict)] = f'{i}_U_shape'
            action_dict[len(action_dict)] = f'{i}_U_shape_prime'
            action_dict[len(action_dict)] = f'{i}_L_shape'
            action_dict[len(action_dict)] = f'{i}_L_shape_prime'
            action_dict[len(action_dict)] = f'{i}_I_shape'
            action_dict[len(action_dict)] = f'{i}_rotation_clockwise_U_shape'
            action_dict[len(action_dict)] = f'{i}_rotation_counterclockwise_U_shape'
            if i != 5:
                action_dict[len(action_dict)] = f'{i}_reimplant_machines_1_to_{i}'
        return action_dict

    def generate_initial_state(self):
        initial_sites = np.array([int(site[0]) for site in self.sites.values()])
        machine_in_site = np.arange(0, self.number_of_machines)
        return np.append(np.concatenate((initial_sites, machine_in_site)), 1)

    def render_interactive_G(self, current_state, cost):
        plt.ion()
        plotting_interactive(self.G, current_state[:self.number_of_machines], cost)
        plt.show()
        plt.clf()
        plt.ioff()

    def positions(self):
        return [coord for key in self.nodes_keys if key.startswith(('P', 'D')) for coord in self.pos[key]]

    def transportation_demand(self):
        transportation = self.transportation_demands.pop(0)
        self.transportation_demands.append(transportation)

        file_path = f'transportation_demand/transportation_demand_M{self.number_of_machines}_{transportation}.csv'
        try:
            self.demand = pd.read_csv(file_path, index_col=0)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            self.demand = pd.DataFrame()

        return self.demand

    def reset(self, render=False):
        file_path = f'transportation_demand/transportation_demand_M{self.number_of_machines}_1.csv'
        self.df = pd.read_csv(file_path, index_col=0)

        self.machine_shape = ['U_shape'] * self.number_of_machines
        initial_state = self.generate_initial_state()
        self.G = self.load_graph().copy()

        config_cost = cost(self.G, self.df)
        initial_cost = config_cost.Total_cost()
        initial_position = self.positions()
        if self.render:
            self.render_interactive_G(initial_state, cost=initial_cost)

        return initial_state, initial_cost, initial_position, self.df

    def step(self, past_state, index_action, past_cost):
        Action = action(self.G, self.sites)
        current_state_array = np.array(past_state.copy())
        action_name = self.action_name_dict[index_action]
        machine_number, shape = action_name.split('_', 1)
        machine_number = int(machine_number)

        common_args = (int(current_state_array[machine_number - 1]), machine_number,
                       int(current_state_array[machine_number + (self.number_of_machines - 1)]))

        if shape.startswith('reimplant_machines'):
            self.handle_reimplant_machines(Action, shape, current_state_array)
        elif shape in ['U_shape', 'U_shape_prime', 'L_shape', 'L_shape_prime', 'I_shape']:
            self.G, machine_state, _, _ = getattr(Action, shape)(*common_args)
            self.machine_shape[machine_number - 1] = shape
            current_state_array[machine_number - 1] = machine_state
        else:
            self.G, machine_state, _, _ = Action.rotation(shape, *common_args)
            current_state_array[machine_number - 1] = machine_state

        self.G = corrector(self.G, self.number_of_machines)
        self.trial_length -= 1

        config_cost = cost(self.G, self.df)
        incentive = getattr(config_cost, self.reward)(past_cost)
        done = self.trial_length <= 0
        new_cost = config_cost.Total_cost()

        if self.render:
            self.render_interactive_G(current_state_array, cost=new_cost)

        if done:
            self.trial_length = self.trial_length_max

        self.H = self.G.copy()
        self.pos = {n: d['pos'] for n, d in self.G.nodes(data=True)}

        return self.G, current_state_array, incentive, done, new_cost, self.positions(), self.df

    def handle_reimplant_machines(self, Action, shape, current_state_array):
        machines_to_swap = [int(num) for num in shape.split('_') if num.isdigit()]
        new_ref_node_1, new_ref_node_2, site_machine1, site_machine2 = getattr(Action, 'reimplant_machines')(
            self.number_of_machines, current_state_array, machines_to_swap[0], machines_to_swap[1])

        self.update_machine_state(Action, new_ref_node_1, machines_to_swap[0], site_machine1, current_state_array)
        self.update_machine_state(Action, new_ref_node_2, machines_to_swap[1], site_machine2, current_state_array)

    def update_machine_state(self, Action, new_ref_node, machine_num, site_machine, current_state_array):
        self.G, machine_state, _, _ = getattr(Action, self.machine_shape[machine_num - 1])(new_ref_node, machine_num,
                                                                                           site_machine)
        current_state_array[machine_num - 1] = machine_state
        current_state_array[machine_num + (self.number_of_machines - 1)] = str(site_machine)

    def close(self):
        plt.close('all')
        print("Environment closed")


# agent = Env(9, 40, 40, 'scaled_cost', [1], render=True)  # Create environment
#
# state, init_cost, pos, _ = agent.reset()  # Reset environment
# for i in range(40):  # Loop 10 times
#     # act = random.randint(0, 7)
#     act = random.randint(0, 62)
#     G, new_state, reward, dn, init_cost, position, _ = agent.step(state, act, init_cost)  # Take step in environment
#     print(new_state)
#     state = new_state  # Update state
