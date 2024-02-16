# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 07:16:44 2024

@author: Alien@System
"""
import operator
import networkx as nx
import re
from collections import defaultdict
import numpy as np


def prepare_annealing(total_set):
    """Preparation function. As the connection graph is the most memory-intensive part, we don't
    build it twice, but prepare it beforehand and serve it to the optimization as a parameter.
    In addition, the word set needs to be cleaned up, as words that are substrings of other words
    might mess things up and eat calculation power. We also reverse the strings, since it's a lot
    easier to build the graph and strings that way.
    The connection graph holds for every pair of strings (A,B) the best way to connect these strings, as two numbers:
        The first is the amount of symbols needed to add that string to the other.
        The second is the position at which to branch off the new string, as a negative index or 0 if we can just append it.
    """
    # Reduce set by removing any substrings
    reduced_set = []
    for entry in sorted(total_set, key=len, reverse=True):
        if not any(entry in parent for parent in reduced_set):
            reduced_set.append(entry)
    # Reverse Entries because it's easier to build that way
    reversed_set = [entry[::-1] for entry in reduced_set]
    # Generate a lookup for the connector weights and insert positions
    graph = {}
    for first in reversed_set:
        for last in reversed_set:
            if first == last:
                continue
            remaining_length = len(last)
            insert_position = 0
            for n in range(1, min(len(first), len(last))):
                if last[:n] in first:
                    insert_position = first.rfind(last[:n]) + n - len(first)
                    remaining_length = len(last) - n
            graph[(first, last)] = (remaining_length, insert_position)
    return reversed_set, graph


def do_swap(old, new, T):
    """Standard form MCMC swapping decider: For T!=0, inefficient swaps are done with an exponential probability decaying
    with T"""
    return new <= old or T != 0 and np.random.rand() < np.exp((old - new)/T)


def anneal_meridians(reduced_set, graph, T):
    """This is a Markov Chain Monte Carlo approach to the minimalisation of the Meridians.
    It works better for the Matchless Kung-Fu use-case than polynomial-time approximations.
    See https://arxiv.org/abs/2210.09986"""
    # Generate a start graph
    working_tree = nx.DiGraph()
    meridian_weight = 6
    while meridian_weight > 5:
        meridian_weight = 1
        working_tree.clear()
        start_perm = np.random.permutation(reduced_set)
        total_length = len(start_perm[0])
        for i in range(1, len(reduced_set)):
            edge = (start_perm[i-1], start_perm[i])
            working_tree.add_edge(*edge)
            insert_pos = graph[edge][1]
            if insert_pos != 0:
                meridian_weight += 1
            total_length += graph[edge][0]

    # Lookup for the edges since we can't call a np.random function direct on graph.keys()
    edges = [edge for edge in graph]
    # Minimizing via Markov chain: Select a random edge to add to the tree, check if this makes the tree shorter
    max_iters = 2 * len(edges)
    curr_tries = 0
    previous_swap = False
    while curr_tries < max_iters:
        new_edge = edges[np.random.randint(len(edges))]
        if new_edge in working_tree.edges:
            continue
        # Get the edge to replace
        if nx.has_path(working_tree, new_edge[1], new_edge[0]):
            # We swap around the node hierarchy, the edge to remove is the parent of the new root
            in_edges = working_tree.in_edges(new_edge[0])
            node_swapping = True
            for parent_node in working_tree.predecessors(new_edge[1]):
                break
            else:
                parent_node = False
        else:
            # We don't swap root, the edge to remove is the parent of the moving node
            in_edges = working_tree.in_edges(new_edge[1])
            node_swapping = False
        for old_edge in in_edges:
            break

        # Ko Rule: Don't undo the swap we just did
        if old_edge == previous_swap:
            curr_tries += 1
            continue

        # Meridian Check: Never go above 5
        # adding an edge:
        def in_change(new_edge):
            if graph[new_edge][1] != 0:
                # Insert type, definitely new meridian
                return 1
            elif any(graph[edge][1] == 0 for edge in working_tree.out_edges(new_edge[0])):
                # Append type, only new meridian if an append type is present at new parent
                return 1
            return 0

        # removing an edge:
        def out_change(old_edge):
            if graph[old_edge][1] != 0:
                # Insert type, always loses a Meridian if removed
                return -1
            elif sum(graph[edge][1] == 0 for edge in working_tree.out_edges(old_edge[0])) > 1:
                # Append type, only loses a Meridian if another append type is present
                return -1
            return 0

        meridian_delta = 0
        meridian_delta += in_change(new_edge)
        meridian_delta += out_change(old_edge)
        # If we do a parent swap, append change of that, too
        if node_swapping and parent_node:
            swap_to_edge = (parent_node, new_edge[0])
            swap_from_edge = (parent_node, new_edge[1])
            meridian_delta += in_change(swap_to_edge)
            meridian_delta += out_change(swap_from_edge)

        if meridian_weight + meridian_delta > 5:
            curr_tries += 1
            continue

        length_delta = graph[new_edge][0] - graph[old_edge][0]
        # If we swap parent, there is extra length-change from that
        if node_swapping and parent_node:
            length_delta += graph[swap_to_edge][0] - graph[swap_from_edge][0]
        # If we change root, there is extra length-change from the base word
        if node_swapping and not parent_node:
            length_delta += len(new_edge[0]) - len(new_edge[1])

        # Based on old and new weight measures, possibly perform a swap
        if do_swap(total_length, total_length + length_delta, T):
            working_tree.remove_edge(*old_edge)
            working_tree.add_edge(*new_edge)
            meridian_weight += meridian_delta
            total_length += length_delta
            if node_swapping and parent_node:
                working_tree.remove_edge(*swap_from_edge)
                working_tree.add_edge(*swap_to_edge)
            previous_swap = new_edge
            curr_tries = 0
        else:
            curr_tries += 1

    return working_tree, total_length, meridian_weight


def multi_anneal(reduced_set, graph, T, attempts):
    """Since Markov Chains are random, multiple executions can produce different results.
    This function performs the annealing operation multiple times and returns the best."""
    results = [anneal_meridians(reduced_set, graph, T) for _ in range(attempts)]
    # Append dipping as extra tie-breaker, with negative values because we want to max it
    results = [(*result, -get_dipping_rates(result[0], graph)) for result in results]
    results = sorted(results, key=operator.itemgetter(1, 2, 3))
    return results[0]


def assemble(word_tree, graph):
    """Function to turn the tree of words into a list of strings that reflect the required meridian layout."""
    reversed_tree = word_tree.reverse(True)
    iter_order = nx.topological_sort(reversed_tree)
    #branch_index = 1
    for work_node in iter_order:
        root_word = work_node
        edges = [(work_node, child) for child in word_tree.successors(work_node)]
        append_edge = False
        for edge in edges:
            if graph[edge][1] == 0:
                append_edge = edge
                edges.remove(edge)
                break
        for edge in edges:
            # Inserts, which leave their child nodes intact
            edge_info = graph[edge]
            insert_root = edge[1]
            cut_at = len(insert_root) - edge_info[0]
            word_tree.nodes[insert_root]['string'] = word_tree.nodes[insert_root]['string'][cut_at:]
            word_tree.nodes[insert_root]['insert_position'] = len(root_word) + edge_info[1]
            for child in word_tree.successors(insert_root):
                if word_tree.nodes[child]['insert_position'] < -edge_info[0]:
                    # If an insert were to be cut away, lift it up to the parent
                    word_tree.add_edge(work_node, child)
                    word_tree.remove_edge(insert_root, child)
                    word_tree.nodes[child]['insert_position'] += len(root_word) + edge_info[1]
                word_tree.nodes[child]['insert_position'] -= edge_info[0]
        if append_edge:
            # Appending, which collapses the child node
            edge_info = graph[append_edge]
            to_remove = append_edge[1]
            cut_at = len(to_remove) - edge_info[0]
            previous_length = len(root_word)
            root_word += word_tree.nodes[to_remove]['string'][cut_at:]
            for child in word_tree.successors(to_remove):
                word_tree.add_edge(work_node, child)
                word_tree.nodes[child]['insert_position'] += previous_length - cut_at
            word_tree.remove_node(to_remove)
        word_tree.nodes[work_node]['string'] = root_word
    branch_index = 1
    for work_node in word_tree:
        root_word = word_tree.nodes[work_node]['string']
        inserts = [(child, word_tree.nodes[child]['insert_position']) for child in word_tree.successors(work_node)]
        # Do them in reverse order to prevent them from affecting the indexing
        inserts = sorted(inserts, key=operator.itemgetter(1), reverse=True)
        for child, insert in inserts:
            in_sign = "%d" % branch_index
            branch_index += 1
            root_word = root_word[:insert] + in_sign + root_word[insert:]
            word_tree.nodes[child]['string'] = in_sign + word_tree.nodes[child]['string']
        word_tree.nodes[work_node]['string'] = root_word

    assembled = [word_tree.nodes[node]['string'] for node in word_tree]
    #Reverse
    assembled = [entry[::-1] for entry in assembled]
    return assembled


def get_dipping_rates(word_tree, graph):
    """Function to get the multi-dipping bonus amount from the hierarchy graph, without having to assemble the string.
    Used as a tiebreaker for the multi_anneal function."""
    # First pass from leaves to root: Get the total length for each node, and who the best one is for passing on the meridians
    for work_node in nx.topological_sort(word_tree.reverse(False)):
        for parent in word_tree.predecessors(work_node):
            word_length = graph[(parent, work_node)][0]
            break
        else:
            word_length = len(work_node)  # The length if it's the root node
        # Minimum meridians needing to go down here
        if word_tree.out_degree(work_node) == 0:
            word_tree.nodes[work_node]['min_dip'] = 1
        elif word_tree.out_degree(work_node) == 1:
            # If we have only one child, but we have to insert it, it's an extra Meridian we need
            for child in word_tree.successors(work_node):
                break
            word_tree.nodes[work_node]['min_dip'] = word_tree.nodes[child]['min_dip'] + (1 if graph[(work_node,child)][1] != 0 else 0)
        else:
            word_tree.nodes[work_node]['min_dip'] = sum(word_tree.nodes[child]['min_dip'] for child in word_tree.successors(work_node))
        child_options = [(child, word_tree.nodes[child]['word_length']) for child in word_tree.successors(work_node)]
        if child_options:
            best_child = max(child_options, key=operator.itemgetter(1))
            word_length += best_child[1]
            word_tree.nodes[work_node]['dip_child'] = best_child[0]
        word_tree.nodes[work_node]['word_length'] = word_length
    # Second pass from root to leaves: Determine Dip bonuses
    dip_total = 0
    for work_node in nx.topological_sort(word_tree):
        for parent in word_tree.predecessors(work_node):
            dip_total += word_tree.nodes[work_node]['dip_strength'] * graph[(parent, work_node)][0]
            break
        else:
            word_tree.nodes[work_node]['dip_strength'] = 5
            dip_total += 5 * len(work_node)
        spare_dip = word_tree.nodes[work_node]['dip_strength'] - word_tree.nodes[work_node]['min_dip']
        # Pass the dip strength on to the kids
        for child in word_tree.successors(work_node):
            if child == word_tree.nodes[work_node]['dip_child']:
                word_tree.nodes[child]['dip_strength'] = word_tree.nodes[child]['min_dip'] + spare_dip
            else:
                word_tree.nodes[child]['dip_strength'] = word_tree.nodes[child]['min_dip']
            # For inserted children, we lose a bit of drip strength from the symbols not used by them
            dip_total += graph[(work_node, child)][1] * word_tree.nodes[child]['dip_strength']
    return dip_total


def get_cost_info(assembled):
    """Helper function to get the total amount of symbols used from an assembled string"""
    totals_string = "".join(assembled)
    totals_string = re.sub(r"[0-9]","",totals_string)
    sign_totals = defaultdict(lambda: 0)
    for sign in totals_string:
        sign_totals[sign] += 1
    dip_strings = assembled.copy()
    for branch_index in range(1,len(assembled)):
        branch_string = "%d" % branch_index
        candidates = [(index, word) for index, word in enumerate(assembled) if branch_string in word]
        branches = [entry for entry in candidates if entry[1][-1:] == branch_string]
        parent = [entry for entry in candidates if entry not in branches][0]
        new_end = dip_strings[parent[0]][dip_strings[parent[0]].index(branch_string)+1:]
        for index, _ in branches:
            dip_strings[index] = dip_strings[index][:-1] + new_end
    for index, string in enumerate(dip_strings):
        # Index clean-up
        dip_strings[index] = re.sub(r"[0-9]", "", string)
    if len(dip_strings) < 5:
        max = sorted(dip_strings, key=len)[-1]
        dip_max_index = dip_strings.index(max)
        dip_strings += [max] * (5 - len(dip_strings))
    else:
        dip_max_index = -1
    dip_totals = "".join(dip_strings)
    dip_sign_totals = defaultdict(lambda: 0)
    for sign in dip_totals:
        dip_sign_totals[sign] += 1
    return totals_string, sign_totals, dip_totals, dip_sign_totals, dip_max_index


def total_set(chi_set):
    """Function to have all entries in an elemental chi set be included in the optimization"""
    return [art[0] for art in chi_set]


def minimal_set(chi_set):
    """Function to reduce an elemental chi set to one giving the best possible bonus for the element
    at minimal cost to the meridians. As the bonus caps at 5, it reduces the set down to a subset that
    sums to 5 (if possible) and has the least total weight and meridian branching when optimized as
    its own layout. In the rare case that the chi_set allows summing to more than 5 but not 5 (e.g.
    two arts with bonus 3) it will take a layout with a smaller bonus for computational efficiency
    (so in the example one art of the two). If this is a problem, use total_set instead."""
    values = [art[1] for art in chi_set]
    if sum(values) <= 5:
        return [art[0] for art in chi_set]

    num_arts = len(chi_set)
    sum_poss = [[False for i in range(6)] for j in range(num_arts)]
    for i in range(num_arts):
        sum_poss[i][0] = True

    if chi_set[0][1] <= 5:
        sum_poss[0][chi_set[0][1]] = True

    for i in range(1, num_arts):
        for j in range(0, 6):
            if chi_set[i][1] <= j:
                sum_poss[i][j] = (sum_poss[i - 1][j] or sum_poss[i - 1][j - chi_set[i][1]])
            else:
                sum_poss[i][j] = sum_poss[i - 1][j]

    goal = 5
    while sum_poss[num_arts - 1][goal] is False and goal > 0:
        goal -= 1
    if goal == 0:
        return []
    res = []
    subsetsum_recursive(chi_set, num_arts - 1, goal, [], res, sum_poss)
    amounts = []
    for subset in res:
        work_set, graph = prepare_annealing(subset)
        _, total_length, meridian_amount = anneal_meridians(work_set, graph, 0)
        amounts.append((subset, total_length, meridian_amount))
    amounts = sorted(amounts, key=operator.itemgetter(1,2))
    return amounts[0][0]


def subsetsum_recursive(chi_set, i, goal, p, res, lookup):
    """Recursive part of the subsetsum calculation used by minimal_set"""
    if i == 0 and goal != 0 and lookup[0][goal]:
        p.append(chi_set[i][0])
        res.append(p)
        return
    if i == 0 and goal == 0:
        res.append(p)
        return
    if lookup[i - 1][goal]:
        b = p.copy()
        subsetsum_recursive(chi_set, i - 1, goal, b, res, lookup)
    if goal >= chi_set[i][1] and lookup[i - 1][goal - chi_set[i][1]]:
        p.append(chi_set[i][0])
        subsetsum_recursive(chi_set, i - 1, goal - chi_set[i][1], p, res, lookup)


def get_boosts_needed(amount):
    """Tiny helper function to turn the number of nodes needed into an Inner Kung-Fu requirement"""
    # Amounts of Signs available:
    #   Base: 18
    #   Primordial Chaos: +12
    #   True Method: +15
    if amount <= 18:
        return "Any"
    if amount <= 30:
        return "Primordial Chaos or True Method"
    if amount <= 33:
        return "True Method"
    if amount <= 45:
        return "Primordial Chaos and True Method"
    return "Impossible"

### --- Begin User-Customizable Block ---

# Signs used are ⟁⧈⭗ in approximation of the spirals they are in the game

# Following the definition of all the Inner Kung-Fu arts, sorted by their Chi type
# Those you don't have or need, comment out to exclude them from the assembly process
sun_set = [
    ('⧈⧈⟁', 1), # Heart-Burning Spell
    ('⧈⟁⭗⧈⧈', 2), # Art of Marrow Refining
    ('⭗⭗⭗⟁⧈⭗', 2), # Purple Seven Star
    ('⧈⟁⭗⧈⟁⭗', 3), # Wu Dang Pure Yang
    ('⟁⭗⭗⧈⧈⟁', 2), # Art of Heartflame
    ]
moon_set = [
    ('⭗⧈⟁', 1), # Freeze Spell
    ('⭗⭗⟁', 1), # Mechanic
    ('⭗⧈⟁⭗⭗⟁', 1),  # Art of Five Senses
    ('⟁⭗⧈⭗⟁', 3), # Insufficient Power
    ('⟁⧈⧈⭗⧈⟁', 2), # Tranquility Spell
    ('⭗⭗⟁⭗⧈⭗', 2), # Stars Divine
    ]
venom_set = [
    ('⧈⧈⟁⭗⟁', 1), # Poison Proof Spell
    ('⭗⟁⧈⟁⭗', 2), # Anti-Toxin Spell
    ]
harmony_set = [
    ('⟁⧈⭗⟁⧈', 1), # Art of Harmony
    ('⭗⧈⟁⭗⟁⟁', 1), # Drunken Immortal
    ('⭗⟁⟁', 1), # Art of Drunkness
    ('⭗⟁⧈⭗⟁⧈', 1), # Yi Jin Jing
    ('⭗⭗⟁⧈', 1), # Lotus in Heart
    ]
misc_set = [
    "⟁⧈⭗⧈⟁",  # Bionic Pentaform
    "⭗⧈⟁⧈⭗⧈",  # Dragonform
    '⭗⟁⧈',  # Breathing Skill
    '⧈⭗⧈⟁⭗⟁',  # Golden Acupuncture
    '⟁⟁⧈⭗⟁', # Art of Hellish Breathing
    '⭗⟁⟁⧈⧈', # Art of Shifting
    '⧈⧈⟁⟁', # Turtle Preservation
    '⧈⟁⭗⭗⟁⧈', # Traceless Heart
    '⟁⧈⟁⧈⟁', # Blood Circulation
    ]

# To reduce the number of signs, we reduce the elemental arts down while retaining the maximum elemental chi bonus
# If you do want to over-fill an elemental chi, use the total_set function instead
mini_sun = minimal_set(sun_set)
mini_moon = minimal_set(moon_set)
mini_venom = minimal_set(venom_set)
mini_harmony = minimal_set(harmony_set)

### --- End User-Customizable Block ---

all_arts = misc_set + mini_sun + mini_moon + mini_venom + mini_harmony
#print(all_arts)

work_set, graph = prepare_annealing(all_arts)
work_tree, _, _, _ = multi_anneal(work_set, graph, 0, 10)
debug_copy = work_tree.copy()
assembled = assemble(work_tree, graph)
totals_string, sign_totals, dip_totals, dip_sign_totals, dip_max_index = get_cost_info(assembled)

meridians = len(assembled)
signs = len(totals_string)
dip_amount = len(dip_totals)

print("Meridians: %d" % len(assembled))
print("Signs: %d (%s, %s)" % (signs, get_boosts_needed(signs), ", ".join("%dx%s" % (sign_totals[sign], sign) for sign in sign_totals)))
print("Dipping Bonuses: %d/%d (%s)" % (dip_amount, 5*signs, ", ".join("%dx%s" % (dip_sign_totals[sign], sign) for sign in dip_sign_totals)))

for i, line in enumerate(assembled):
    print(("*" if i == dip_max_index else "") + line)
