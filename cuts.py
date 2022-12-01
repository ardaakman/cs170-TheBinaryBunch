#Import external libraries and starter code
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
from starter import *

#Import built in libraries
import random

#Algorithm to initially group out the nodes into zero weight groups.
def find_groups(G):
    groups = []
    visited = set()
    for node in G.nodes:
        if node not in visited:
            group = set()
            group.add(node)
            visited.add(node)
            for node2 in G.nodes:
                if node2 not in visited and not G.has_edge(node, node2):
                    group.add(node2)
                    visited.add(node2)
            groups.append(group)
    return groups


"""
This is an attempt to solving the problem with a min cut approach. The inputs to the function are:
    G: The graph
    num_teams: The number of teams we want to split the graph into. This will be an estimate of the teams, due to how diff_param works.
    diff_param: The difference parameter. This is a float between 0 and 1. The difference parameter is used to determine the upper and lower bound for the size of the groups.
    evens: This is a boolean value. If true, the function will not add a node to a group, if that would make the node go over the treshold.
"""
def min_cut_sol(G, num_teams, diff_param, evens = False):
    #Count for debugging purposes
    count = 0
    #Evens parameter ensures, when trying to split into groups the smallest group loop does not give its node to another group if that would put it above the largest
    #group treshold, even if this is the optimal solution.
    groups = find_groups(G)
    num_nodes = len(G.nodes)
    #Set upper bound and lower bound based on diff_param
    upper_bound_size = num_nodes//num_teams + (num_nodes * diff_param)
    lower_bound_size = num_nodes//num_teams - (num_nodes * diff_param)
    #Jai is weird with rough_size.
    largest_group = max(groups, key=len)
    smallest_group = min(groups, key=len)
    largest_group_size = len(largest_group)
    smallest_group_size = len(smallest_group)
    #While the largest group is larger than the upper bound or the smallest group is smaller than the lower bound
    #Outer loop for the case that the below while loop triggers a group to have size larger than largest groups size.
    old_group = None
    while largest_group_size > upper_bound_size or smallest_group_size < lower_bound_size:
        count+=1
        if count % 1000 == 0:
            old_group = groups
            print(groups)
            print("\n\n")
        if old_group == groups:
            break
        if largest_group_size > upper_bound_size:
            #If the largest group is larger than the upper bound, remove the edge with the lowest min cut with another group.
            #Overall smallest_group. This represent the group that gives smallest value for all different node/group combinations.
            change_group, smallest_node = None, None
            for node in largest_group:
                #Calculate the cut value of the node with the current group.
                curr_cut = 0
                for node2 in largest_group:
                    if node2 == node:
                        continue
                    weight = G.get_edge_data(node, node2, 0)
                    if weight == 0:
                        weight = 0
                    else:
                        weight = weight['weight']
                    curr_cut += weight
                #Assign the min_group value here. 
                min_group = None
                min_val = float('inf')
                #Here, we loop through every single other group and check the nodes in each group. 
                #We try to find the group that gives the smallest cut for the given node we have at the moment.
                for other in groups:
                    if other == largest_group:
                        continue
                    total = 0
                    for node2 in other:
                        weight = G.get_edge_data(node, node2, 0)
                        if weight == 0:
                            weight = 0
                        else:
                            weight = weight['weight']
                        if weight == float('inf'):
                            weight = 0
                        total += weight
                    #Here, total becomes the cut value of the node with the other group - the cut value of the node with the current group.
                    total = total - curr_cut
                    if min_val > total and len(other) < upper_bound_size:
                        min_val = total
                        min_group = other
                if min_group != None:
                    change_group, smallest_node = min_group, node
            #If we find a group that gives us a smaller cut, we move the node to that group.
            if change_group != None:
                largest_group.remove(smallest_node)
                change_group.add(smallest_node)
                largest_group, smallest_group = max(groups, key=len), min(groups, key=len)
                largest_group_size, smallest_group_size = len(largest_group), len(smallest_group)
        #Nothing means no changes could be made in the below largest_group_size if condition.
        else:
            count+= 1
            #If the smallest group is smaller than the lower bound, remove nodes one by one to the groups that give the minimum cut.
            change_group, smallest_node = None, None
            for node in smallest_group:
                min_group = None
                min_val = float('inf')
                for other in groups:
                    if other == smallest_group:
                        continue
                    total = 0
                    for node2 in other:
                        weight = G.get_edge_data(node, node2, 0)
                        if weight == 0:
                            weight = 0
                        else:
                            weight = weight['weight']
                        total += weight
                    if min_val > total:
                        min_val = total
                        min_group = other
                if min_group != None:
                    change_group, smallest_node = min_group, node
            if change_group != None:
                smallest_group.remove(smallest_node)
                change_group.add(smallest_node)
                largest_group, smallest_group = max(groups, key=len), min(groups, key=len)
                largest_group_size, smallest_group_size = len(largest_group), len(smallest_group)
                #If we have completly emptied a group, remove that group from the list of groups.
                if smallest_group_size == 0:
                    groups.remove(smallest_group)
                    smallest_group = min(groups, key=len)
                    smallest_group_size = len(smallest_group)
    print(groups)
    for node in G.nodes:
        for group in groups:
            if node in group:
                G.nodes[node]['team'] = groups.index(group)
                break
    print(G)
    return


def min_cut_2(G, num_teams, diff_param) -> None:
    #Count for debugging purposes
    count = 0
    #Evens parameter ensures, when trying to split into groups the smallest group loop does not give its node to another group if that would put it above the largest
    #group treshold, even if this is the optimal solution.
    groups = find_groups(G)
    num_nodes = len(G.nodes)
    #Set upper bound and lower bound based on diff_param
    avg = math.ceil(num_nodes//num_teams)
    upper_bound_size = num_nodes//num_teams + (num_nodes * diff_param)
    lower_bound_size = num_nodes//num_teams - (num_nodes * diff_param)
    #Jai is weird with rough_size.
    largest_group = max(groups, key=len)
    smallest_group = min(groups, key=len)
    largest_group_size = len(largest_group)
    smallest_group_size = len(smallest_group)
    #While the largest group is larger than the upper bound or the smallest group is smaller than the lower bound
    #Outer loop for the case that the below while loop triggers a group to have size larger than largest groups size.
    final_groups = [set() for i in range(num_teams)]
    groups_sorted = sorted(groups, key=len)
    i = 0
    while(i < num_teams):
        curr_group = groups_sorted[0]
        if len(final_groups[0]) == 0:
            break
        if len(curr_group) <= avg:
            final_groups[0].update(curr_group)
            final_groups.sort(key=len)
            groups_sorted.pop(0)
        else:
            break
    
    for group in groups_sorted:
        G = nx.Graph()
        for other in final_groups:
                if len(other) == avg:
                    continue
                length = len(other)
                added = avg - length
                total = 0
                for node2 in other:
                    weight = G.get_edge_data(node, node2, 0)
                    total += weight
                

            


def min_cut_3(G, num_teams, diff_param) -> None: #The one with simple matching
    #Count for debugging purposes
    count = 0
    #Evens parameter ensures, when trying to split into groups the smallest group loop does not give its node to another group if that would put it above the largest
    #group treshold, even if this is the optimal solution.
    groups = find_groups(G)
    num_nodes = len(G.nodes)
    #Set upper bound and lower bound based on diff_param
    avg = math.ceil(num_nodes//num_teams)
    upper_bound_size = num_nodes//num_teams + (num_nodes * diff_param)
    lower_bound_size = num_nodes//num_teams - (num_nodes * diff_param)
    #Jai is weird with rough_size.
    largest_group = max(groups, key=len)
    smallest_group = min(groups, key=len)
    largest_group_size = len(largest_group)
    smallest_group_size = len(smallest_group)
    #While the largest group is larger than the upper bound or the smallest group is smaller than the lower bound
    #Outer loop for the case that the below while loop triggers a group to have size larger than largest groups size.
    final_groups = [set() for i in range(num_teams)]
    groups_sorted = sorted(groups, key=len)
    i = 0
    while(i < num_teams):
        curr_group = groups_sorted[0]
        if len(final_groups[0]) == 0:
            break
        if len(curr_group) <= avg:
            final_groups[0].update(curr_group)
            final_groups.sort(key=len)
            groups_sorted.pop(0)
        else:
            break
    
    for group in groups_sorted:
        chosen_node = None
        chosen_group = None
        for node in group:
            min_val = float("inf")
            for other in final_groups:
                if len(other) == avg:
                    continue
                total = 0
                for node2 in other:
                    weight = G.get_edge_data(node, node2, 0)
                    total += weight
                if total < min_val:
                    chosen_node = node
                    chosen_group = other
        if chosen_group != None:
            group.remove(chosen_node)
            chosen_group.add(chosen_node)
        else:
            break
    
        

                

            


    


#def min_cut_sol_attempt_2(G, num_teams, diff_param):
#This will be similar to min weight, but now use cut values instead.
def test(size, diff):
    G = read_input(r'inputs/small21.in')
    min_cut_sol(G, size, diff, False)
    print(score(G))


    
    

