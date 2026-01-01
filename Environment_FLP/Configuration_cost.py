import networkx as nx
from networkx.classes.function import path_weight
import numpy as np
from pulp import *
import pandas as pd


class cost:
    def __init__(self, graph, df):
        self.graph = graph
        self.df = df
        # self.df.set_index('pickup\delivery', inplace=True)

    def shortest_path(self):
        # Extract pickup and delivery points from self.df
        pickup = [key for key in self.df.index if key.startswith('P')]
        delivery = [key for key in self.df.columns if key.startswith('D')]

        flow = []
        for delivery_point in delivery:
            for pickup_point in pickup:
                try:
                    path = nx.dijkstra_path(self.graph, delivery_point, pickup_point, weight='weight')
                    path_length = path_weight(self.graph, path, weight="weight")
                    flow.append(path_length)
                except nx.NetworkXNoPath:
                    flow.append(float('inf'))  # If no path is found, use infinity as the distance
        distance_matrix = np.reshape(flow, (len(pickup), len(delivery)))
        return distance_matrix

    def loaded_cost(self):
        a = self.shortest_path()
        b = self.df.to_numpy()
        # print(a)
        # print('=============')
        # print(b)
        # print('loaded cost:\n', np.sum(np.multiply(a.transpose(), b)))
        return np.sum(np.multiply(a.transpose(), b))

    def empty_cost(self):
        # Creates the 'prob' variable to contain the problem data
        prob = LpProblem("Supply_Problem", LpMinimize)

        # Creates a list of all the supply nodes
        warehouses = self.df.columns.values

        # Creates a dictionary for the number of units  of supply for each supply node
        supply = {str(i): self.df[str(i)].sum() for i in self.df.columns.values}

        # Creates a list of all demand nodes
        projects = self.df.index.values

        # Creates a dictionary for the number of units of demand for each demand node
        demand = self.df.sum(axis=1)

        # # Creates a list of costs of each transportation path
        costs = self.shortest_path()

        # The cost data is made into a dictionary
        costs = makeDict([warehouses, projects], costs, 0)

        # Creates a list of tuples containing all the possible routes for transport
        Routes = [(w, b) for w in warehouses for b in projects]

        # A dictionary called 'Vars' is created to contain the referenced variables(the routes)
        vars = LpVariable.dicts("Route", (warehouses, projects), 0, None, LpInteger)

        # The minimum objective function is added to 'prob' first
        prob += (
            lpSum([vars[w][b] * costs[w][b] for (w, b) in Routes])
        )

        # The supply maximum constraints are added to prob for each supply node (warehouses)
        for w in warehouses:
            prob += (
                    lpSum([vars[w][b] for b in projects]) == supply[w])

        # The demand minimum constraints are added to prob for each demand node (project)
        for b in projects:
            prob += (
                    lpSum([vars[w][b] for w in warehouses]) == demand[b])

        # The problem is solved using PuLP's choice of Solver
        prob.solve(PULP_CBC_CMD(msg=False))

        # Print the variables optimized value
        # for v in prob.variables():
        #     print(v.name, "=", v.varValue)

        # The optimised objective function value is printed to the screen
        empty_cost = value(prob.objective)
        return empty_cost

    def Total_cost(self):
        loaded_cost = self.loaded_cost()
        empty_cost = self.empty_cost()
        # print('===================================================================')
        # print('Total_cost of this configuration is:')
        # print('Empty_cost:\n', empty_cost)
        # print('Total_cost:\n', loaded_cost + empty_cost)
        return empty_cost + loaded_cost

    def inverse_cost(self, old_cost):
        total_cost = self.Total_cost()
        if total_cost <= old_cost:
            deviation = 1/self.Total_cost()
        else:
            deviation = -1
        return deviation

    def difference_cost(self, old_cost):
        total_cost = self.Total_cost()
        if total_cost <= old_cost:
            return (old_cost-self.Total_cost())*0.01
        else:
            deviation = -1
            return deviation

    def scaled_cost(self, old_cost):
        total_cost = self.Total_cost()
        if total_cost <= old_cost:
            reward = (1750-total_cost)/(1750-400)
        else:
            reward = -1
        return reward
