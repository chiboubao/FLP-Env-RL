from itertools import product
import networkx as nx
from collections import OrderedDict


def calculate_distance(pickup, delivery):
    distance = abs(pickup[0] - delivery[0]) + abs(pickup[1] - delivery[1])
    return distance


class action:

    def __init__(self, G, sites):
        self.G = G
        self.sites = sites
        self.pos = {n: d['pos'] for n, d in self.G.nodes(data=True)}

    def calculate_pos_I(self, a, b):
        x_1, y_1 = self.pos[a]
        x_2, y_2 = self.pos[b]
        if x_1 - x_2 == 0:
            x = x_1
        else:
            x = (x_1 + x_2) * 0.5
        if y_1 - y_2 == 0:
            y = y_1
        else:
            y = (y_1 + y_2) * 0.5
        return x, y

    def calculate_pos_U(self, a, b):
        x_1, y_1 = self.pos[a]
        x_2, y_2 = self.pos[b]

        if x_1 - x_2 == 0:
            x = x_1
            t = x_1
        elif x_1 - x_2 < 0:
            x = x_1 + 1
            t = x_1 + 2
        else:
            x = x_1 - 1
            t = x_1 - 2

        if y_1 - y_2 == 0:
            y = y_1
            z = y_1

        elif y_1 - y_2 < 0:
            y = y_1 + 1
            z = y_1 + 2

        else:
            y = y_1 - 1
            z = y_1 - 2
        return x, y, t, z

    def calculate_pos_L(self, a, b, c):
        x_1, y_1 = self.pos[a]
        x_2, y_2 = self.pos[b]
        x_3, y_3 = self.pos[c]
        if x_1 == x_2:
            x = x_1
            y = (y_1 + y_2) * 0.5
        else:
            x = (x_1 + x_2) * 0.5
            y = y_1
        if x_2 == x_3:
            t = x_2
            z = (y_2 + y_3) * 0.5
        else:
            t = (x_2 + x_3) * 0.5
            z = y_2
        return x, y, t, z

    def find_site(self, reference_nodes):
        list_of_sites = list(self.sites.values())
        for i in range(0, len(list_of_sites)):
            if set(reference_nodes).issubset(list_of_sites[i]):
                s = list(self.sites)[i]
                return s

    def I_shape(self, a, Machine_ID, site):
        a = str(a)
        nodes_in_sites = list(self.sites.values())
        site1 = nodes_in_sites[site]
        index = site1.index(a)
        a, b, c, d = site1[index:] + site1[:index]
        a, b, c, d = str(a), str(b), str(c), str(d)
        x, y = self.calculate_pos_I(a, b)
        t, z = self.calculate_pos_I(c, d)
        # Machine_ID = self.find_site(nodes)
        pickup = 'P' + str(Machine_ID)
        delivery = 'D' + str(Machine_ID)
        pickup_delivery = [pickup, delivery]
        self.G.remove_nodes_from(pickup_delivery)
        self.pos[pickup] = [x, y]
        self.pos[delivery] = [t, z]
        self.G.add_nodes_from([(pickup, {'pos': self.pos[pickup]}),
                               (delivery, {'pos': self.pos[delivery]})])
        self.G.add_edge(a, delivery, weight=calculate_distance(self.pos[delivery], self.pos[a]))
        self.G.add_edge(delivery, b, weight=calculate_distance(self.pos[delivery], self.pos[b]))
        self.G.add_edge(c, pickup, weight=calculate_distance(self.pos[c], self.pos[pickup]))
        self.G.add_edge(pickup, d, weight=calculate_distance(self.pos[pickup], self.pos[d]))
        return self.G, int(a), Machine_ID, site

    def U_shape_prime(self, a, Machine_ID, site):
        a = str(a)
        nodes_in_sites = list(self.sites.values())
        site1 = nodes_in_sites[site]
        index = site1.index(a)
        a, b, c, d = site1[index:] + site1[:index]
        a, b, c, d = str(a), str(b), str(c), str(d)
        x, y, t, z = self.calculate_pos_U(a, b)
        # Machine_ID = self.find_site(nodes)
        pickup = 'P' + str(Machine_ID)
        delivery = 'D' + str(Machine_ID)
        pickup_delivery = [pickup, delivery]
        self.G.remove_nodes_from(pickup_delivery)
        self.pos[delivery] = [x, y]
        self.pos[pickup] = [t, z]
        self.G.add_nodes_from(self.pos.keys())
        self.G.add_nodes_from([(pickup, {'pos': self.pos[pickup]}),
                               (delivery, {'pos': self.pos[delivery]})])
        self.G.add_edge(a, delivery, weight=calculate_distance(self.pos[a], self.pos[delivery]))
        self.G.add_edge(delivery, pickup, weight=calculate_distance(self.pos[delivery], self.pos[pickup]))
        self.G.add_edge(pickup, b, weight=calculate_distance(self.pos[pickup], self.pos[b]))
        return self.G, int(a), Machine_ID, site

    def U_shape(self, a, Machine_ID, site):
        a = str(a)
        nodes_in_sites = list(self.sites.values())
        site1 = nodes_in_sites[site]
        index = site1.index(a)
        a, b, c, d = site1[index:] + site1[:index]
        a, b, c, d = str(a), str(b), str(c), str(d)
        x, y, t, z = self.calculate_pos_U(a, b)
        pickup = 'P' + str(Machine_ID)
        delivery = 'D' + str(Machine_ID)
        pickup_delivery = [pickup, delivery]
        self.G.remove_nodes_from(pickup_delivery)
        self.pos[pickup] = [x, y]
        self.pos[delivery] = [t, z]
        self.G.add_nodes_from([(pickup, {'pos': self.pos[pickup]}),
                               (delivery, {'pos': self.pos[delivery]})])
        self.G.add_nodes_from([(pickup, {'pos': self.pos[pickup]}),
                               (delivery, {'pos': self.pos[delivery]})])
        self.G.add_edge(a, pickup, weight=calculate_distance(self.pos[pickup], self.pos[delivery]))
        self.G.add_edge(pickup, delivery, weight=calculate_distance(self.pos[pickup], self.pos[delivery]))
        self.G.add_edge(delivery, b, weight=calculate_distance(self.pos[pickup], self.pos[delivery]))
        return self.G, int(a), Machine_ID, site

    def L_shape(self, a, Machine_ID, site):
        a = str(a)
        nodes_in_sites = list(self.sites.values())
        site1 = nodes_in_sites[site]
        index = site1.index(a)
        a, b, c, d = site1[index:] + site1[:index]
        a, b, c, d = str(a), str(b), str(c), str(d)
        x, y, t, z = self.calculate_pos_L(a, b, c)
        # Machine_ID = self.find_site(nodes)
        pickup = 'P' + str(Machine_ID)
        delivery = 'D' + str(Machine_ID)
        pickup_delivery = [pickup, delivery]
        self.G.remove_nodes_from(pickup_delivery)
        self.pos[pickup] = [x, y]
        self.pos[delivery] = [t, z]
        self.G.add_nodes_from(self.pos.keys())
        self.G.add_nodes_from([(pickup, {'pos': self.pos[pickup]}),
                               (delivery, {'pos': self.pos[delivery]})])
        self.G.add_edge(a, pickup, weight=calculate_distance(self.pos[a], self.pos[pickup]))
        self.G.add_edge(pickup, b, weight=calculate_distance(self.pos[pickup], self.pos[b]))
        self.G.add_edge(b, delivery, weight=calculate_distance(self.pos[b], self.pos[delivery]))
        self.G.add_edge(delivery, c, weight=calculate_distance(self.pos[delivery], self.pos[c]))
        return self.G, int(a), Machine_ID, site

    def L_shape_prime(self, a, Machine_ID, site):
        a = str(a)
        nodes_in_sites = list(self.sites.values())
        site1 = nodes_in_sites[site]
        index = site1.index(a)
        a, b, c, d = site1[index:] + site1[:index]
        a, b, c, d = str(a), str(b), str(c), str(d)
        x, y, t, z = self.calculate_pos_L(a, b, c)
        # Machine_ID = self.find_site(nodes)
        pickup = 'P' + str(Machine_ID)
        delivery = 'D' + str(Machine_ID)
        pickup_delivery = [pickup, delivery]
        self.G.remove_nodes_from(pickup_delivery)
        self.pos[delivery] = [x, y]
        self.pos[pickup] = [t, z]
        self.G.add_nodes_from(self.pos.keys())
        self.G.add_nodes_from([(pickup, {'pos': self.pos[pickup]}),
                               (delivery, {'pos': self.pos[delivery]})])
        self.G.add_edge(a, delivery, weight=calculate_distance(self.pos[a], self.pos[delivery]))
        self.G.add_edge(delivery, b, weight=calculate_distance(self.pos[b], self.pos[delivery]))
        self.G.add_edge(b, pickup, weight=calculate_distance(self.pos[b], self.pos[pickup]))
        self.G.add_edge(pickup, c, weight=calculate_distance(self.pos[pickup], self.pos[c]))
        return self.G, int(a), Machine_ID, site

    @staticmethod
    def get_shape_type(function_name):
        # Split the function name by underscores
        parts = function_name.split('_')
        # Extract the shape type and rotation type
        rotation_type = parts[1]
        shape_info = parts[2:]
        shape_type = '_'.join(shape_info)
        return rotation_type, shape_type

    def rotation(self, function_name, a, Machine_ID, site):
        rotation_type, shape = self.get_shape_type(function_name)
        # print('shape:', shape)
        self.G, a, Machine_ID, site = getattr(self, f"rotation_{rotation_type}")(a, Machine_ID, site, shape)
        return self.G, a, Machine_ID, site

    def rotation_clockwise(self, a, Machine_ID, site, shape):
        a = str(a)
        nodes_in_sites = list(self.sites.values())
        site1 = nodes_in_sites[site]
        index = site1.index(a)
        x, y, z, t = site1[index:] + site1[:index]
        a, b, c, d = str(y), str(z), str(t), str(x)
        getattr(self, shape)(a, Machine_ID, site)
        return self.G, int(a), Machine_ID, site

    def rotation_counterclockwise(self, a, Machine_ID, site, shape):
        a = str(a)
        nodes_in_sites = list(self.sites.values())
        site1 = nodes_in_sites[site]
        index = site1.index(a)
        x, y, z, t = site1[index:] + site1[:index]
        a, b, c, d = str(t), str(x), str(y), str(z)
        getattr(self, shape)(a, Machine_ID, site)
        return self.G, int(a), Machine_ID, site

    def change_machine(self, a, Machine_ID, site):
        num_machines = 4  # Total number of machines
        new_machine_id = (Machine_ID % num_machines) + 1
        return new_machine_id

    def connected_point_machine(self, machine):
        point_connected_machine = list(nx.neighbors(self.G, 'P' + str(machine))) + list(
            nx.neighbors(self.G, 'D' + str(machine)))
        connected_points_machine = [point for point in point_connected_machine if point.isdigit()]
        # remove redundant points within connected points
        connected_points_machine = list(OrderedDict.fromkeys(connected_points_machine))
        # print('connected_points_1:', connected_points_machine)
        return connected_points_machine

    def find_new_reference_node(self, node1, node2, site1, site2):
        nodes_in_sites = list(self.sites.values())

        # Convert numpy.int32 to Python int
        site1 = int(site1)
        site2 = int(site2)

        # print(f"Converted site1: {site1}, site2: {site2}")
        # print(f"Number of available sites: {len(nodes_in_sites)}")

        # Adjust range check to allow 0 if it's valid
        if site1 < 0 or site1 >= len(nodes_in_sites) or site2 < 0 or site2 >= len(nodes_in_sites):
            raise ValueError(
                f"site1 ({site1}) or site2 ({site2}) is out of valid range. Valid range is 0 to {len(nodes_in_sites) - 1}.")

        # If inputs are 0-based, no need to subtract 1
        site_1 = nodes_in_sites[site1]
        site_2 = nodes_in_sites[site2]

        # print(f"Site {site1}: {site_1}")
        # print(f"Site {site2}: {site_2}")

        if str(node1) not in site_1 or str(node2) not in site_2:
            raise ValueError(f"node1 ({node1}) or node2 ({node2}) not found in the corresponding site.")

        index_reference_node_1 = site_1.index(str(node1))
        index_reference_node_2 = site_2.index(str(node2))

        new_reference_node1 = site_2[index_reference_node_1]
        new_reference_node2 = site_1[index_reference_node_2]

        return new_reference_node1, new_reference_node2, site1, site2

    def reimplant_machines(self, number_of_machines, state, machine1, machine2):
        # print('machine1 -1:', machine1 - 1)
        # print('state', state)
        # print('(current_state[machine1 -1]):', state[machine1 - 1])

        reference_node_1 = state[machine1-1]
        reference_node_2 = state[machine2-1]
        site_1 = state[machine1 + (number_of_machines-1)]
        site_2 = state[machine2 + (number_of_machines-1)]
        new_ref_node_1, new_ref_node_2, site_machine1, site_machine2 = self.find_new_reference_node(
            reference_node_1, reference_node_2, site_1, site_2)
        site_1, site_2 = site_2, site_1
        return new_ref_node_1, new_ref_node_2, site_1, site_2


