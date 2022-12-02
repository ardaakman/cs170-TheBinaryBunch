import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
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
                graph.nodes[node]['team'] = i
                break
    return


def preprocess(graph: nx.Graph, num_partitions: int, imbalance_tolerance: int) -> typing.Tuple[nx.Graph, List[List[int]]]:
    graph = graph.copy()
    num_original_nodes = graph.number_of_nodes()
    num_new_nodes = num_original_nodes + imbalance_tolerance
    graph.add_nodes_from(range(num_original_nodes, num_new_nodes))
    nodes = list(range(num_new_nodes))
    random.shuffle(nodes)
    num_per_partition = math.ceil(num_new_nodes//num_partitions)
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
        for node_a, d_a in node_d_from_a:
            for node_b, d_b in node_d_from_b:
                change = d_a + d_b - 2*graph.get_edge_data(node_a, node_b, {"weight": 0})["weight"]
                if change < min_change and node_a not in locked_a and node_b not in locked_b:
                    min_change = change
                    change_from_a = node_a
                    change_from_b = node_b
        partition_a_curr.remove(change_from_a)
        partition_a_curr.append(change_from_b)
        partition_b_curr.remove(change_from_b)
        partition_b_curr.append(change_from_a)
        node_d_from_a, node_d_from_b = compute_d(graph, partition_a_curr, partition_b_curr)
        locked_a.add(change_from_a)
        locked_b.add(change_from_b)
        gains.append(min_change)

    k, min_sum, curr_sum = (0, 0, 0)
    for i in range(len(gains)):
        curr_sum += gains[i]
        if curr_sum < min_sum:
            min_sum = curr_sum
            k = i + 1
    for i in range(k):
        partition_a.append(locked_a[i])
        partition_a.remove(locked_b[i])
        partition_b.append(locked_b[i])
        partition_b.remove(locked_a[i])
    return




def compute_d(graph: nx.Graph, partition_a: List[int], partition_b: List[int]) -> Tuple[Dict[int, int], Dict[int, int]]:
    node_d_from_a = {}
    node_d_from_b = {}
    for node in partition_a:
        node_d_from_a[node] = 0
        node_d_from_a[node] -= node_weight(graph, node, partition_b)
        node_d_from_a[node] += node_weight(graph, node, partition_a)
    for node in partition_b:
        node_d_from_b[node] = 0
        node_d_from_b[node] -= node_weight(graph, node, partition_a)
        node_d_from_b[node] += node_weight(graph, node, partition_b)
    return node_d_from_a, node_d_from_b


# Helper functions
def node_weight(graph: nx.Graph, node: int, group: List[int]) -> int:
    to_return = 0
    for other_node in group:
        to_return += graph.get_edge_data(node, other_node, {"weight": 0})["weight"]
    return to_return


def test(x):
    G = nx.Graph()
    G.add_nodes_from(range(10))
    G = preprocess(G, 3, 2)
    return G.nodes[x]["team"]

