import networkx as nx
import numpy as np

def mychoice(list):
    return list[np.random.randint(len(list))]

def diff(a, b):
    """ 
        Return set difference of lists a,b as list
    """
    b = set(b)
    return [aa for aa in a if aa not in b]

def get_trace(v, parentmap):
    ret = []
    while v != None:
        ret.append(v)
        v = parentmap[v]
    return ret

def dfs_recursive(G, node, visited, forbidden, min_length, mode_weight, temp_forbidden = set([])):
    """ 
        Do a recursive depth first search
            G               - graph
            node            - node to start at
            visited         - list of previously visited nodes
            forbidden       - nodes that can't be in cycle
            min_length      - minimal cycle length
            mode_weight     - probab. to continue in same mode
            temp_forbidden  - used to temporarily exclude cycles that are too short
    """

    allowed_nodes = diff(G.successors(node), forbidden.union(temp_forbidden))

    if len(allowed_nodes) == 0:
        # Reached leaf
        return False

    # choose random successor
    if len(visited) >= 2 and len(allowed_nodes) >= 2:
        current_mode = G[visited[-2]][visited[-1]]['modes'][0]
        next_modes = list(set().union(*[G[node][next_node]['modes']
                                        for next_node in allowed_nodes]))
        if current_mode in next_modes:
            # 
            ind = next_modes.index(current_mode)
            next_node_same = allowed_nodes[ind]
            if np.random.random(1) < mode_weight:
                # stick with same mode
                next_node = next_node_same
            else:
                # choose other mode uniformly
                next_node = mychoice(diff(allowed_nodes, set([next_node_same])))
        else:
            # can not continue with current mode
            next_node = mychoice(allowed_nodes) 
    else:
        next_node = mychoice(allowed_nodes)

    if next_node in visited:
        # found a cycle
        cycle = visited[ visited.index(next_node): ]

        if len(cycle) > min_length:
            # rotate it
            rot_ind = np.argmin(cycle)
            return cycle[rot_ind:] + cycle[:rot_ind]
        else:
            # must choose other node
            return dfs_recursive(G, node, visited, forbidden, min_length, mode_weight, set([next_node]))

    # no cycle, continue down graph
    test = dfs_recursive(G, next_node, visited + [next_node], forbidden, min_length, mode_weight)
    if test:
        # found a cycle downstream
        return test 
    else:
        # need to pick another vertex!
        return dfs_recursive(G, node, visited, forbidden.union(set([next_node])), min_length, mode_weight)


def dfs(G, initnode, forbidden, min_length, mode_weight):
    """ 
        Do a depth first search
            G               - graph
            node            - node to start at
            visited         - list of previously visited nodes
            forbidden       - nodes that can't be in cycle
            min_length      - minimal cycle length
            mode_weight     - probab. to continue in same mode
            temp_forbidden  - used to temporarily exclude cycles that are too short
    """

    stack = []
    stack.append(initnode)

    visited = { node : False for node in G.nodes_iter() }
    visited[initnode] = True

    parentmap = { initnode : None }
    
    while len(stack) > 0:

        current_node = stack.pop()
        visited[current_node] = True

        successors = G.successors(current_node)
        valid_successors = []

        trace = get_trace(current_node, parentmap)

        for successor in successors:
            if successor in trace:
                # look for cycle
                cycle = trace[:trace.index(successor)+1]
                if len(cycle) >= min_length:
                    return list(reversed(cycle))
                else:
                    # cycle was too short, keep looking
                    continue
            elif not visited[successor] and successor not in forbidden:
                # add to stack/parentmap
                parentmap[successor] = current_node
                valid_successors.append(successor)

        # add children to current_node
        if len(trace) >= 2:
            current_mode = G[trace[1]][trace[0]]['modes'][0]
        else:
            current_mode = -1

        # add successors to stack (put current mode on top with higher proba)
        np.random.shuffle(valid_successors)
        successor_modes = [G[current_node][next_node]['modes'][0] for next_node in valid_successors ]
        if current_mode in successor_modes:
            # should add edge with current mode on top of stack
            # with proba mode_weight
            next_node_same = valid_successors[successor_modes.index(current_mode)]
            if np.random.random(1) < mode_weight:
                # stick with same mode
                stack.extend( diff( valid_successors, [next_node_same] ))
                stack.append(next_node_same)
            else:
                # choose other mode uniformly
                stack.append(next_node_same)
                stack.extend( diff( valid_successors, [next_node_same] ) )
        else:
            # can not use same mode, continue with random mode
            stack.extend( valid_successors )

    return None # No cycle was found starting at that point!


def random_cycle(G, forbidden = [], min_length = 2, mode_weight = 0.5, recursive = True):
    '''
    Generate a random simple cycle in the graph G

    Inputs:
      G           : graph to search in
                     class: networkx DiGraph
      forbidden   : nodes that can't be in cycle
      min_length  : minimal cycle length to consider
                     type: int
      mode_weight : when randomly selecting successors, select same mode with 
                    this probability
                     type: double in interval [0,1]
    Returns:
      cycle       : list of nodes in G
      False       : if no simple cycle longer than min_length exists

    Comments: - exhaustive search can be slow, best for graphs with an abundance of cycles
              - algorithm is a randomized DFS 
              - the probability for a given cycle to be selected is unknown and is not uniform
    '''
    cycle = False
    init_nodes = G.nodes()
    np.random.shuffle(init_nodes)
    while not cycle: 
        if len(init_nodes) == 0:
            return False
        init_node = init_nodes.pop()
        if recursive:
            cycle = dfs_recursive(G, init_node, [init_node], set([]), min_length, mode_weight)
        else:
            cycle = dfs(G, init_node, forbidden, min_length, mode_weight)
    verify_cycle(G, cycle)
    return cycle


def verify_cycle(G, cyc):
    for i in range(len(cyc)):
        this_node = cyc[i]
        next_node = cyc[(i+1) % len(cyc)]
        assert(next_node in G.successors(this_node))