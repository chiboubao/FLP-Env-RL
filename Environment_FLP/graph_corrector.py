from itertools import product
import networkx as nx


def corrector(G, N):
    pos_dict = {n: d['pos'] for n, d in G.nodes(data=True)}
    for y_axis in range(0, N*3+1, 3):
    # for y_axis in range(0, ):
        # print(y_axis)
        nodes = [k for k, v in pos_dict.items() if v[0] == y_axis]
        for node in nodes:
            edges_to_remove = [(node, neighbor) for neighbor in G.neighbors(node)]
            G.remove_edges_from(edges_to_remove)

        # combination of all possible edges between nodes
        # combos = product(nodes, repeat=2)
        # combos = list(combos)
        # print('combos', combos)
        # # remove the nodes
        # G.remove_edges_from(combos)

        # plotting(H, pos)
        # sorted dict for the nodes by descending order
        position = [v for k, v in pos_dict.items() if k in nodes]
        nodes_to_modify = dict(zip(nodes, position))
        sortedList_of_pos = sorted(nodes_to_modify.values(), reverse=True)
        # print('sortedList_of_pos:', sortedList_of_pos)
        sorted_nodes_to_modify = {}
        for sortedKey in sortedList_of_pos:
            for key, value in nodes_to_modify.items():
                if value == sortedKey:
                    sorted_nodes_to_modify[key] = value
        # print('sorted_nodes_to_modify', sorted_nodes_to_modify)
        nodes = [k for k, v in sorted_nodes_to_modify.items() if v[0] == y_axis]
        edges = list(map(list, zip(nodes, nodes[1:])))
        # print('edges to add:', edges)
        weights = []
        for k, v in zip(sortedList_of_pos, sortedList_of_pos[1:]):
            weights.append(k[1] - v[1])
        # print('weights:', weights)
        for i, j in zip(edges, weights):
            G.add_edge(i[0], i[1], weight=j)

    ##########################################################################
    for x_axis in range(0, N*3+1, 3):
        nodes = [k for k, v in pos_dict.items() if v[1] == x_axis]
        # combination of all possible edges between nodes
        combos = product(nodes, repeat=2)
        combos = list(combos)
        # remove the nodes
        G.remove_edges_from(combos)
        # plotting(H, pos)
        # sorted dict for the nodes by descending order
        position = [v for k, v in pos_dict.items() if k in nodes]
        nodes_to_modify = dict(zip(nodes, position))
        sortedList_of_pos = sorted(nodes_to_modify.values(), reverse=True)
        # print('sortedList_of_pos:', sortedList_of_pos)
        sorted_nodes_to_modify = {}
        for sortedKey in sortedList_of_pos:
            for key, value in nodes_to_modify.items():
                if value == sortedKey:
                    sorted_nodes_to_modify[key] = value
        # print('sorted_nodes_to_modify', sorted_nodes_to_modify)
        nodes = [k for k, v in sorted_nodes_to_modify.items() if v[1] == x_axis]
        edges = list(map(list, zip(nodes, nodes[1:])))
        # print('edges to add:', edges)
        weights = []
        for k, v in zip(sortedList_of_pos, sortedList_of_pos[1:]):
            weights.append(k[0] - v[0])
        # print('weights:', weights)
        for i, j in zip(edges, weights):
            G.add_edge(i[0], i[1], weight=j)

    return G