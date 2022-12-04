#Import external libraries and starter code
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
from starter import *
import networkx.algorithms.bipartite as bipartite


#Import built in libraries
import random
from collections import defaultdict

#Algorithm to initially group out the nodes into zero weight groups.
def find_groups(G):
    groups = []
    visited = set()
    coloring = nx.greedy_color(G)
    group_nos = set()
    for val in coloring.values():
        if val in group_nos:
            continue
        else:
            group_nos.add(val)
    group_dict = defaultdict(list)
    for no in coloring:
        if not(coloring[no] in group_dict):
            group_dict[coloring[no]] = [no]
        else:
            group_dict[coloring[no]].append(no)
    sets = [set(arr) for arr in group_dict.values()]
    return sets

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
    for node in G.nodes:
        for group in groups:
            if node in group:
                G.nodes[node]['team'] = groups.index(group)
                break
    return


def min_cut_2(G, num_teams, diff_param) -> None:
    #Count for debugging purposes
    count = 0
    #Evens parameter ensures, when trying to split into groups the smallest group loop does not give its node to another group if that would put it above the largest
    #group treshold, even if this is the optimal solution.
    groups = initial_groups = find_groups(G)
    print(initial_groups)
    num_nodes = len(G.nodes)
    #Set upper bound and lower bound based on diff_param
    avg = math.ceil(num_nodes//num_teams)
    if (diff_param == 0):
        upper_bound_size = avg
        lower_bound_size = avg
    else:
        upper_bound_size = int(num_nodes//num_teams + math.ceil(num_nodes * diff_param))
        lower_bound_size = int(num_nodes//num_teams - math.ceil(num_nodes * diff_param))
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
    # print("initial groups", groups_sorted)
    # print("\n")
    while(i < num_teams):
        if len(groups_sorted) == 0:
            break
        curr_group = groups_sorted[0]
        if len(curr_group) <= avg:
            final_groups[0].update(curr_group)
            final_groups.sort(key=len)
            groups_sorted.pop(0)
        else:
            i += 1
    # print("After initial addition:", final_groups)
    # print("\n")
    #Deal with the rest of the values, using bipartitie matching.
    for group in groups_sorted:
        final_groups = match_bipartite(G, group, final_groups, upper_bound_size)
    # print("Final Groups", final_groups)
    # print("\n Initial Groups", initial_groups)


    for node in G.nodes:
        for group in final_groups:
            if node in group:
                G.nodes[node]['team'] = final_groups.index(group) + 1
                break
    return

def calculate_cut(G, other_group, curr):
    total = 0
    for other_node in other_group:
        total += G.get_edge_data(curr, other_node, {"weight": 0})["weight"]
    return total


def match_bipartite(G, group, final_groups, avg):
    trial_G = nx.Graph()
    group_map = defaultdict(list) #Mapping group nos to the no of the corresponding graph nodes in the graph.
    group_reverse = {}
    i = 0
    k = 0
    for other in final_groups:
        length = len(other)
        added = avg - length
        # print(added)
        for _ in range(added):
            group_map[k].append(i)
            group_reverse[i] = k
            i += 1
        k += 1
    # So the mapping becomes (group_map) - {0: [0, 1, 2, 3, 4], 1: [5, 6, 7, 8]} etc.
    for m in group_reverse.keys():
        trial_G.add_node(m)
    
    #This part saves the original node values, so we can both calculate the edge weights and also do the conversion 
    #after the bipartite matching.
    #First one is original val to k.
    original_mapping = {}
    #Second one is k to original val.
    original_reversed = {}
    # print("AAAAAA\n")
    # print(i)
    # print("\n")
    for val in group:
        original_mapping[val] = i
        original_reversed[i] = val
        i+=1

    #Now, we have min cut values for the node combinations.
    for node in group:
        #Calculate cut value with the group.
        trial_G.add_node(original_mapping[node])
        for other in final_groups:
            cut_val = calculate_cut(G, other, node)
            #Add copies
            for adding in group_map[final_groups.index(other)]:
                trial_G.add_edge(original_mapping[node], adding, weight=cut_val)
    #Now, we have the graph, we can do the matching.
    # print("\n", "Group Map")
    # print(group_map)
    # print("\n", "Group Reverse")
    # print(group_reverse)
    # print("\n", "Original Mapping")
    # print(original_mapping)
    # print("\n", "Original Reversed")
    # print(original_reversed)
    
    # print("AFTER HERE OMIT")
    # print("\n", trial_G.nodes)
    # print("\n", original_mapping.values())
    matching = nx.algorithms.bipartite.minimum_weight_full_matching(trial_G, original_reversed.keys())
    #Matching is so that node i to j is matching[i] == j.
    # print("\n", "Matching")
    # print(matching)
    # print("\n")
    val_set = set()
    count = 0
    for matched in matching.keys():
        if matched in val_set:
            continue
        count+=1
        # print(matched, matching[matched])
        true_node = original_reversed[matched]
        # print("Count", count)
        val_set.add(matching[matched])
        true_group = group_reverse[matching[matched]]
        final_groups[true_group].add(true_node)
    #This should deal with one whole group - move to the next one!
    return final_groups
    



    



            

            


def min_cut_3(G, num_teams, diff_param) -> None: #The one with simple matching
   #Count for debugging purposes
    count = 0
    #Evens parameter ensures, when trying to split into groups the smallest group loop does not give its node to another group if that would put it above the largest
    #group treshold, even if this is the optimal solution.
    groups = initial_groups = find_groups(G)
    print(initial_groups)
    num_nodes = len(G.nodes)
    #Set upper bound and lower bound based on diff_param
    avg = math.ceil(num_nodes//num_teams)
    if (diff_param == 0):
        upper_bound_size = avg
        lower_bound_size = avg
    else:
        upper_bound_size = int(num_nodes//num_teams + math.ceil(num_nodes * diff_param))
        lower_bound_size = int(num_nodes//num_teams - math.ceil(num_nodes * diff_param))
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
    # print("initial groups", groups_sorted)
    # print("\n")
    while(i < num_teams):
        if len(groups_sorted) == 0 or len():
            break
        curr_group = groups_sorted[0]
        if len(curr_group) <= avg:
            final_groups[0].update(curr_group)
            final_groups.sort(key=len)
            groups_sorted.pop(0)
        else:
            i += 1
    # print("After initial addition:", final_groups)
    # print("\n")
    #Deal with the rest of the values, using bipartitie matching.
    for group in groups_sorted:
        final_groups = match_bipartite(G, group, final_groups, upper_bound_size)
    # print("Final Groups", final_groups)
    # print("\n Initial Groups", initial_groups)


    for node in G.nodes:
        for group in final_groups:
            if node in group:
                G.nodes[node]['team'] = final_groups.index(group) + 1
                break
    return

def shitty_input(G, num_teams):
    groups = find_groups(G)
    print(groups)
    print("Group count", len(groups), "\n\n")
    print(G.edges)
    for node in groups[0]:
        for node2 in groups[0]:
            if node == node2:
                continue
            if (G.get_edge_data(node, node2, {"weight": 0})['weight']) != 0:
                print("danger")
                print(G.get_edge_data(list(groups[0])[i], list(groups[0])[node], {"weight": 0})['weight'])
    sum_val = sum([len(g) for g in groups])
    if len(groups) == 1:
        #Just split all to groups
        new_groups = []
        rest = [set() for i in range(num_teams)]
        for node in G.nodes:
            rest[0].add(node)
            rest.sort(key= len)
        new_groups.extend(rest)
        for node in G.nodes:
            for group in new_groups:
                if node in group:
                    G.nodes[node]['team'] = new_groups.index(group) + 1
                    break
        return
    max_group = max(groups, key=len)
    min_group = min(groups, key=len)
    if len(groups) > 2:
        for node in G.nodes:
            for group in groups:
                if node in group:
                    G.nodes[node]['team'] = groups.index(group) + 1
                    break
        return
    while(len(min_group) < math.floor(sum_val//num_teams)):
        if len(groups) > 2:
            break
        min_elem, min_val = None, float("inf")
        for elem in max_group:
            value = calculate_cut(G, min_group, elem)
            if value < min_val:
                min_val = value
                min_elem = elem
        max_group.remove(min_elem)
        min_group.add(min_elem)
    #When done with doing the above, divide the rest into smaller ones.
    division = sum_val//num_teams
    new_groups = [min_group]
    rest = [set() for i in range(num_teams-1)]
    for node in G.nodes:
        if node in min_group:
            continue
        rest[0].add(node)
        rest.sort(key= len)
    new_groups.extend(rest)
    total_cover = 0
    for g in new_groups:
        total_cover += len(g)
    for node in G.nodes:
        for group in new_groups:
            if node in group:
                G.nodes[node]['team'] = new_groups.index(group) + 1
                break
    return


        

def solver(G):
    best_score = float("inf")
    best_graph = None
    G_copied = G.copy()
    for teams in range(5, 25):
        for params in np.arange(0.02, 0.1, 0.01):
            min_cut_2(G_copied, teams, params)
            curr_score = score(G_copied)
            if curr_score < best_score:
                best_score = curr_score
                best_graph = G_copied.copy()
    for i in range(len(G.nodes)):
        G.nodes[i]["team"] = int(best_graph.nodes[i]["team"])
    return best_graph

def solver_shitty(G):
    best_score = float("inf")
    best_graph = None
    G_copied = G.copy()
    for teams in range(2, 20):
        min_cut_2(G_copied, teams, 0.01)
        curr_score = score(G_copied)
        if curr_score < best_score:
            best_score = curr_score
            best_graph = G_copied.copy()
    for i in range(len(G.nodes)):
        G.nodes[i]["team"] = int(best_graph.nodes[i]["team"])
    return best_graph



#def min_cut_sol_attempt_2(G, num_teams, diff_param):
#This will be similar to min weight, but now use cut values instead.
def run_2():
    run_all(solver, 'smallinputs', 'outputs', overwrite=True)
    tar('outputs', overwrite=True)


    
def test(input, num_teams, diff):
    G = read_input(r'inputs/{}.in'.format(input))
    min_cut_2(G, num_teams, 0.05)
    validate_output(G)
    print(score(G))
    write_output(G, r'outputs/{}.out'.format(input), overwrite=True)

<<<<<<< HEAD
tar('outputs', overwrite=True)
=======
tar('outputs', overwrite=True)
>>>>>>> d631cfcfae30bb288ca2f65a75db42c6c255a17b
