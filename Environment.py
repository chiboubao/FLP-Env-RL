import random
import time
import json
from itertools import combinations

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from Environment_FLP.plotting import plotting_interactive
from Environment_FLP.Actions import action
from Environment_FLP.Configuration_cost import cost
from Environment_FLP.graph_corrector import corrector
from Environment_FLP.canonical import LayoutGeometry


def convert_keys_to_integers(d):
    return {int(key): value for key, value in d.items()}


class Env:
    """
    State vector layout (unchanged): [reference node of machine 1..N, site index of machine 1..N, 1]

    Parameters added on top of the original ones:
        canonical:  if True, after every step (and at reset) the layout is replaced by its canonical
                    representative under the rotations/mirrors of the workshop grid. The cost is
                    identical, but the agent only ever sees one state per equivalence class.
        reward_bounds: (min_cost, max_cost) used by the 'scaled_cost' reward. None means
                    (400, 1750) for 4 workstations (values of the paper) and (0, initial cost of
                    the demand) otherwise.
        invalid_action_reward: reward given when an action would put two P/D points on the same
                    spot; the action is then cancelled (the layout does not change).
    """

    def __init__(self, number_of_machines, trial_length, trial_length_max, reward, transportation_demands, render,
                 canonical=True, reward_bounds=None, invalid_action_reward=-1):
        self.number_of_machines = number_of_machines
        self.trial_length = trial_length
        self.trial_length_max = trial_length_max
        self.reward = reward
        self.transportation_demands = transportation_demands
        self.canonical = canonical
        self.reward_bounds = reward_bounds
        self.invalid_action_reward = invalid_action_reward

        self.reset_counter = 0
        self.current_episode = 0
        self.state_visit_counter = {}
        self.invalid_action_count = 0

        self.random_generator = np.random.default_rng(int(time.time()))
        self.machine_shape = ['U_shape'] * self.number_of_machines

        self.G = self.load_graph()
        self.H = self.G.copy()
        self.pos = {n: d['pos'] for n, d in self.G.nodes(data=True)}

        self.sites = self.load_sites()
        self.geometry = LayoutGeometry(self.sites, self.pos)
        self.nodes_keys = self.generate_node_keys()
        self.action_name_dict = self.generate_action_dict()

        self.observation_space = self.number_of_machines * 4 + self.number_of_machines ** 2
        self._action_space = len(self.action_name_dict)

        self.transport = None
        self.demand = None
        self.df = None
        self.cost_bounds = None

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
        """
        Per workstation: 5 shapes + 2 rotations (the rotation keeps the current shape).
        Then one reimplantation action per unordered pair {m, n}, m != n
        (N(N-1)/2 actions: 6 for 4 workstations, 36 for 9, 120 for 16).
        """
        action_dict = {}
        for i in range(1, self.number_of_machines + 1):
            action_dict[len(action_dict)] = f'{i}_U_shape'
            action_dict[len(action_dict)] = f'{i}_U_shape_prime'
            action_dict[len(action_dict)] = f'{i}_L_shape'
            action_dict[len(action_dict)] = f'{i}_L_shape_prime'
            action_dict[len(action_dict)] = f'{i}_I_shape'
            action_dict[len(action_dict)] = f'{i}_rotation_clockwise'
            action_dict[len(action_dict)] = f'{i}_rotation_counterclockwise'
        for m, n in combinations(range(1, self.number_of_machines + 1), 2):
            action_dict[len(action_dict)] = f'{m}_reimplant_machines_{m}_to_{n}'
        return action_dict

    def generate_initial_state(self):
        initial_sites = np.array([int(site[0]) for site in self.sites.values()])
        machine_in_site = np.arange(0, self.number_of_machines)
        return np.append(np.concatenate((initial_sites, machine_in_site)), 1)

    # ------------------------------------------------------------------ layout <-> state
    def state_to_layout(self, state):
        n = self.number_of_machines
        return [(int(state[n + m]), str(int(state[m])), self.machine_shape[m]) for m in range(n)]

    def layout_to_state(self, layout, template_state):
        n = self.number_of_machines
        state = np.array(template_state).copy()
        for m, (site_index, ref, shape) in enumerate(layout):
            state[m] = int(ref)
            state[n + m] = int(site_index)
            self.machine_shape[m] = shape
        return state

    def build_graph(self, layout):
        """Graph of the workshop for a layout, rebuilt from the P/D positions."""
        G = self.load_graph().copy()
        G.remove_edges_from(list(G.edges()))
        for node, p in self.geometry.pd_positions(layout).items():
            G.nodes[node]['pos'] = [p[0], p[1]]
        return corrector(G, self.number_of_machines)

    def canonicalize(self, state):
        layout, _ = self.geometry.canonical_layout(self.state_to_layout(state))
        state = self.layout_to_state(layout, state)
        return state, self.build_graph(layout)

    # ------------------------------------------------------------------------- render
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

        self.trial_length = self.trial_length_max
        self.machine_shape = ['U_shape'] * self.number_of_machines
        initial_state = self.generate_initial_state()
        self.G = self.load_graph().copy()
        if self.canonical:
            initial_state, self.G = self.canonicalize(initial_state)
        self.pos = {n: d['pos'] for n, d in self.G.nodes(data=True)}

        config_cost = cost(self.G, self.df)
        initial_cost = config_cost.Total_cost()
        if self.reward_bounds is not None:
            self.cost_bounds = tuple(self.reward_bounds)
        elif self.number_of_machines == 4:
            self.cost_bounds = (400, 1750)
        else:
            self.cost_bounds = (0, initial_cost)

        initial_position = self.positions()
        if self.render:
            self.render_interactive_G(initial_state, cost=initial_cost)

        return initial_state, initial_cost, initial_position, self.df

    def step(self, past_state, index_action, past_cost):
        previous_G = self.G.copy()
        previous_shapes = list(self.machine_shape)

        Action = action(self.G, self.sites)
        current_state_array = np.array(past_state).copy()
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
            # rotation keeps the current shape of the workstation
            rotation_name = f'{shape}_{self.machine_shape[machine_number - 1]}'
            self.G, machine_state, _, _ = Action.rotation(rotation_name, *common_args)
            current_state_array[machine_number - 1] = machine_state

        self.trial_length -= 1
        done = self.trial_length <= 0

        layout = self.state_to_layout(current_state_array)
        if self.geometry.has_overlap(self.geometry.pd_positions(layout)):
            # Two P/D points would share the same spot: the action is cancelled.
            self.invalid_action_count += 1
            self.G = previous_G
            self.machine_shape = previous_shapes
            current_state_array = np.array(past_state).copy()
            new_cost = past_cost
            incentive = self.invalid_action_reward
        else:
            if self.canonical:
                current_state_array, self.G = self.canonicalize(current_state_array)
            else:
                self.G = self.build_graph(layout)
            config_cost = cost(self.G, self.df)
            new_cost = config_cost.Total_cost()
            min_cost, max_cost = self.cost_bounds
            incentive = getattr(config_cost, self.reward)(past_cost, total_cost=new_cost,
                                                          min_cost=min_cost, max_cost=max_cost)

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
        current_state_array[machine_num + (self.number_of_machines - 1)] = int(site_machine)

    def close(self):
        plt.close('all')
        print("Environment closed")


if __name__ == '__main__':
    agent = Env(9, 40, 40, 'scaled_cost', [1], render=False)
    state, init_cost, pos, _ = agent.reset()
    for i in range(40):
        act = random.randint(0, len(agent.action_name_dict) - 1)
        G, new_state, reward, dn, init_cost, position, _ = agent.step(state, act, init_cost)
        print(agent.action_name_dict[act], new_state, reward, init_cost)
        state = new_state
