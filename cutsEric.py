import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx

import starter
from starter import *
import networkx.algorithms.bipartite as bipartite
import random
import typing
from typing import Tuple, List, Dict


def kernighan_lin_solution(graph: nx.Graph, num_partitions: int, imbalance_tolerance: int) -> None:
    preprocessed_graph, partitions = preprocess(graph, num_partitions, imbalance_tolerance)
    for i in range(num_partitions):
        for j in range(num_partitions):
            if i == j:
                continue
            kernighan_lin_algorithm(preprocessed_graph, partitions[i], partitions[j])
    for node in graph.nodes:
        for i in range(num_partitions):
            if node in partitions[i]:
                graph.nodes[node]['team'] = i + 1
                break
    return

def kerninghan_lin_grid_solution(graph, num_partitions_min, num_partitions_max, imbalance_tolerance_min, imbalance_tolerance_max):
    min_score = float("inf")
    min_solution = None
    for num_partitions in range(num_partitions_min, num_partitions_max):
        for imbalance_tolerance in range(imbalance_tolerance_min, imbalance_tolerance_max):
            copied_graph = graph.copy()
            kernighan_lin_solution(copied_graph, num_partitions, imbalance_tolerance)
            current_score = score(copied_graph)
            if min_score > current_score:
                min_score = current_score
                min_solution = copied_graph
    for node in graph.nodes:
        graph.nodes[node]["team"] = min_solution.nodes[node]["team"]
    return


def preprocess(graph: nx.Graph, num_partitions: int, imbalance_tolerance: int) -> typing.Tuple[nx.Graph, List[List[int]]]:
    graph = graph.copy()
    num_original_nodes = graph.number_of_nodes()
    num_new_nodes = num_original_nodes + imbalance_tolerance
    graph.add_nodes_from(range(num_original_nodes, num_new_nodes))
    nodes = list(range(num_new_nodes))
    random.shuffle(nodes)
    num_per_partition = math.ceil(num_new_nodes/num_partitions)
    partitions = []
    for partition in range(num_partitions):
        i = 0
        partitions.append([])
        while i < num_per_partition and num_per_partition*partition + i < num_new_nodes:
            partitions[-1].append(nodes[num_per_partition*partition + i])
            i += 1
    return graph, partitions


def kernighan_lin_algorithm(graph: nx.Graph, partition_a: List[int], partition_b: List[int]) -> None:
    node_d_from_a, node_d_from_b = compute_d(graph, partition_a, partition_b)
    partition_a_curr = partition_a.copy()
    partition_b_curr = partition_b.copy()
    locked_a = []
    locked_b = []
    gains = []
    smallest_partition_size = min(len(partition_a), len(partition_b))
    for i in range(smallest_partition_size):
        min_change = float("inf")
        change_from_a = None
        change_from_b = None
        for node_a, d_a in node_d_from_a.items():
            for node_b, d_b in node_d_from_b.items():
                change = d_a + d_b #- 2*graph.get_edge_data(node_a, node_b, {"weight": 0})["weight"]
                if change < min_change and node_a not in locked_a and node_b not in locked_b:
                    min_change = change
                    change_from_a = node_a
                    change_from_b = node_b
        partition_a_curr.remove(change_from_a)
        partition_a_curr.append(change_from_b)
        partition_b_curr.remove(change_from_b)
        partition_b_curr.append(change_from_a)
        node_d_from_a, node_d_from_b = compute_d(graph, partition_a_curr, partition_b_curr)
        locked_a.append(change_from_a)
        locked_b.append(change_from_b)
        gains.append(min_change)

    k, min_sum, curr_sum = (0, 0, 0)
    for i in range(len(gains)):
        curr_sum += gains[i]
        if curr_sum < min_sum:
            min_sum = curr_sum
            k = i + 1
    for i in range(k):
        if locked_b[i] not in partition_a:
            partition_a.append(locked_b[i])
        if locked_a[i] in partition_a:
            partition_a.remove(locked_a[i])
        if locked_a[i] not in partition_b:
            partition_b.append(locked_a[i])
        if locked_b[i] in partition_b:
            partition_b.remove(locked_b[i])
    return


def compute_d(graph: nx.Graph, partition_a: List[int], partition_b: List[int]) -> Tuple[Dict[int, int], Dict[int, int]]:
    node_d_from_a = {}
    node_d_from_b = {}
    for node in partition_a:
        node_d_from_a[node] = 0
        node_d_from_a[node] += node_weight(graph, node, partition_b)
        node_d_from_a[node] -= node_weight(graph, node, partition_a)
    for node in partition_b:
        node_d_from_b[node] = 0
        node_d_from_b[node] += node_weight(graph, node, partition_a)
        node_d_from_b[node] -= node_weight(graph, node, partition_b)
    return node_d_from_a, node_d_from_b


# Helper functions
def node_weight(graph: nx.Graph, node: int, group: List[int]) -> int:
    to_return = 0
    for other_node in group:
        to_return += graph.get_edge_data(node, other_node, {"weight": 0})["weight"]
    return to_return


def test_preprocess(num_partitions=2, imbalance_factor=3):
    g = nx.Graph()
    g.add_nodes_from(range(10))
    new_g, partitions = preprocess(g, num_partitions, imbalance_factor)
    print("Number of Nodes: ", new_g.number_of_nodes())
    print("Length of Partitions: ", sum([len(partition) for partition in partitions]))
    print(partitions)
    return


def test_compute_d():
    G = nx.Graph()
    G.add_nodes_from(range(10))
    G, partitions = preprocess(G, 2, 2)
    return compute_d(G, partitions[0], partitions[1])


def test_algorithm():
    G = nx.Graph()
    G.add_nodes_from(range(4))
    G, partitions = preprocess(G, 2, 0)
    for i in range(4):
        for j in range(4):
            if i != j and not G.has_edge(i, j):
                weight = random.randint(0, 10)
                G.add_edge(i, j, weight=weight)
                print("Weight ", (i, j), ": ", weight)

    print("Initial Partition: ", partitions)
    kernighan_lin_algorithm(G, partitions[0], partitions[1])
    print("Final Partition: ", partitions)
    print("Length of Partition A: ", len(partitions[0]))
    print("Length of Partition B: ", len(partitions[1]))
    return


def test_overall():
    g = nx.Graph()
    g.add_nodes_from(range(4))
    for i in range(4):
        for j in range(4):
            if i != j:
                g.add_edge(i, j, weight=random.randint(1, 10))
    g_input = g.copy()
    g, partitions = preprocess(g, 2, 0)
    for i in range(4):
        if i in partitions[0]:
            g.nodes[i]["team"] = 0
        else:
            g.nodes[i]["team"] = 1
    print("Initial Score: ", score(g))
    kernighan_lin_solution(g_input, 2, 3)
    print("Final Score: ", score(g_input))
    return

def test(input):
    G = read_input(r'inputs/{}.in'.format(input))
    kerninghan_lin_grid_solution(G, 1, 20, 0, 10)
    validate_output(G)
    print(score(G))
    write_output(G, r'outputs/{}.out'.format(input), overwrite=True)


def try_output(first, last):
    for input_file in ["small" + str(i) for i in range(first, last)]:
        test(input_file)
    starter.tar('outputs', overwrite=True)
    return
