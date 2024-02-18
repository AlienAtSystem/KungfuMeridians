# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 07:16:44 2024

@author: Alien@System
"""
import operator
import networkx as nx
import re
from collections import defaultdict
import random
import math
import itertools


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
    # Generate a lookup for the connector weights and branching positions
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
    return new <= old or T != 0 and random.random() < math.exp((old - new)/T)


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
        start_perm = random.sample(reduced_set,len(reduced_set))
        total_length = len(start_perm[0])
        for i in range(1, len(reduced_set)):
            edge = (start_perm[i-1], start_perm[i])
            working_tree.add_edge(*edge)
            insert_pos = graph[edge][1]
            if insert_pos != 0:
                meridian_weight += 1
            total_length += graph[edge][0]

    # Lookup for the edges since we shouldn't call a random function direct on graph.keys()
    edges = [edge for edge in graph]
    # Minimizing via Markov chain: Select a random edge to add to the tree, check if this makes the tree shorter
    max_iters = 2 * len(edges)
    curr_tries = 0
    previous_swap = False
    while curr_tries < max_iters:
        new_edge = random.choice(edges)
        if new_edge in working_tree.edges:
            continue
        # Get the edge to replace
        if nx.has_path(working_tree, new_edge[1], new_edge[0]):
            # We swap around the node hierarchy, the edge to remove is the parent of the new root
            old_edge = (next(working_tree.predecessors(new_edge[0])), new_edge[0])
            node_swapping = True
            for parent_node in working_tree.predecessors(new_edge[1]):
                break
            else:
                parent_node = False
        else:
            # We don't swap root, the edge to remove is the parent of the moving node
            old_edge = (next(working_tree.predecessors(new_edge[1])), new_edge[1])
            node_swapping = False

        # Ko Rule: Don't undo the swap we just did
        if old_edge == previous_swap:
            curr_tries += 1
            continue

        # Meridian Check: Never go above 5
        # adding an edge:
        def in_change(new_edge):
            if graph[new_edge][1] != 0:
                # Branching type, definitely new meridian
                return 1
            elif any(graph[edge][1] == 0 for edge in working_tree.out_edges(new_edge[0])):
                # Append type, only new meridian if an append type is present at new parent
                return 1
            return 0

        # removing an edge:
        def out_change(old_edge):
            if graph[old_edge][1] != 0:
                # Branching type, always loses a Meridian if removed
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
    # The 'branch at this position' logic is nice for the MCMC, but hell for the assembly and dipping calculation
    # Thus, we clean up our tree
    clean_tree(working_tree, graph)
    return working_tree, total_length, meridian_weight


def clean_tree(word_tree, graph):
    """Function to remove the branching edges from the string hierarchy, which makes all following operations easier.
    It works in-place, and since it will change the node structure, it bakes the still required information into the
    nodes themselves."""
    reversed_tree = word_tree.reverse(True)
    iter_order = nx.topological_sort(reversed_tree)
    placeholders = itertools.count(0)
    for work_node in iter_order:
        for parent in word_tree.predecessors(work_node):
            # By default, our base string is the extra characters we need to append to our parent
            edge = graph[(parent, work_node)]
            base_string = work_node[-edge[0]:]
            word_tree.nodes[work_node]['to_remove'] = -edge[1]
            break
        else:
            # Root node, it gets its word as base string
            base_string = work_node
        # Now we adjust the base string based on the child nodes to remove branches
        to_lift = [child for child in word_tree.successors(work_node) if word_tree.nodes[child]['to_remove'] > len(base_string)]
        for child in to_lift:
            # If we'd have to cut more than we're long, lift to our parent to maybe cut from there
            for parent in word_tree.predecessors(work_node):
                word_tree.add_edge(parent, child)
                word_tree.remove_edge(work_node, child)
                word_tree.nodes[child]['to_remove'] -= len(base_string) - word_tree.nodes[work_node]['to_remove']
        # We cut our base string into chunks, each ending at where a branch needs to be appended
        cuts_dict = defaultdict(list)
        for child in word_tree.successors(work_node):
            cuts_dict[word_tree.nodes[child]['to_remove']].append(child)
        cuts_at = list(cuts_dict.keys())
        if 0 not in cuts_at:
            cuts_at.append(0)
        cuts_at = sorted(cuts_at, reverse=True)
        current_parent = work_node
        # Pairwise iteration, by hand because itertools.pairwise is only python 3.10+
        start_it, end_it = itertools.tee(cuts_at)
        next(end_it, None)
        for start, end in zip(start_it, end_it):
            # Now we turn each chunk into a node and append the branches
            new_node = next(placeholders)
            word_tree.add_node(new_node, string=base_string[-start: -end if end else None])
            word_tree.add_edge(current_parent, new_node)
            for child in cuts_dict[end]:
                word_tree.add_edge(new_node, child)
                word_tree.remove_edge(work_node, child)
            current_parent = new_node
        # Cut down our base string
        if cuts_at[0] > 0:
            base_string = base_string[:-cuts_at[0]]
        word_tree.nodes[work_node]['string'] = base_string


def multi_anneal(reduced_set, graph, T, attempts):
    """Since Markov Chains are random, multiple executions can produce different results.
    This function performs the annealing operation multiple times and returns the best."""
    results = [anneal_meridians(reduced_set, graph, T) for _ in range(attempts)]
    # Append dipping as extra tie-breaker, with negative values because we want to max it
    results = [(*result, -get_dipping_rates(result[0])) for result in results]
    results = sorted(results, key=operator.itemgetter(1, 2, 3))
    return results[0]


def assemble(word_tree):
    """Function to turn the cleaned tree of words into a list of strings that reflect the required meridian layout."""
    assembled = []
    branches = itertools.count(1)
    for work_node in nx.topological_sort(word_tree):
        if word_tree.out_degree(work_node) == 0:
            assembled.append(word_tree.nodes[work_node]['string'])
        elif word_tree.out_degree(work_node) == 1:
            for child in word_tree.successors(work_node):
                word_tree.nodes[child]['string'] = word_tree.nodes[work_node]['string'] + word_tree.nodes[child]['string']
        else:
            favored_child = word_tree.nodes[work_node].get('dip_child', next(word_tree.successors(work_node)))
            branch_string = "%d" % next(branches)
            for child in word_tree.successors(work_node):
                if child == favored_child:
                    word_tree.nodes[child]['string'] = (word_tree.nodes[work_node]['string'] + branch_string
                                                        + word_tree.nodes[child]['string'])
                else:
                    word_tree.nodes[child]['string'] = branch_string + word_tree.nodes[child]['string']
    #Reverse
    assembled = [entry[::-1] for entry in assembled]
    return assembled


def get_dipping_rates(word_tree):
    """Function to get the multi-dipping bonus amount from the cleaned hierarchy graph, without having to assemble.
    Used as a tiebreaker for the multi_anneal function."""
    # First pass from leaves to root: Get the total length for each node, and who the best one is for passing on the meridians
    for work_node in nx.topological_sort(word_tree.reverse(False)):
        word_length = len(word_tree.nodes[work_node]['string'])
        # Minimum meridians needing to go down here
        if word_tree.out_degree(work_node) == 0:
            word_tree.nodes[work_node]['min_dip'] = 1
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
        if word_tree.in_degree(work_node) == 0:
            word_tree.nodes[work_node]['dip_strength'] = 5
        dip_total += word_tree.nodes[work_node]['dip_strength'] * len(word_tree.nodes[work_node]['string'])
        spare_dip = word_tree.nodes[work_node]['dip_strength'] - word_tree.nodes[work_node]['min_dip']
        # Pass the dip strength on to the kids
        for child in word_tree.successors(work_node):
            if child == word_tree.nodes[work_node]['dip_child']:
                word_tree.nodes[child]['dip_strength'] = word_tree.nodes[child]['min_dip'] + spare_dip
            else:
                word_tree.nodes[child]['dip_strength'] = word_tree.nodes[child]['min_dip']
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
        candidates = [(index, word) for index, word in enumerate(dip_strings) if branch_string in word]
        if candidates:
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
    return min(amounts, key=operator.itemgetter(1,2))[0]


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


# --- Begin User-Customizable Block ---

# Signs used are ⟁⧈⭗ in approximation of the spirals they are in the game

# Following the definition of all the Inner Kung-Fu arts, sorted by their Chi type
# Those you don't have or need, comment out to exclude them from the assembly process
sun_set = [
    ('⧈⧈⟁', 1),  # Heart-Burning Spell
    ('⧈⟁⭗⧈⧈', 2),  # Art of Marrow Refining
    ('⭗⭗⭗⟁⧈⭗', 2),  # Purple Seven Star
    ('⧈⟁⭗⧈⟁⭗', 3),  # Wu Dang Pure Yang
    ('⟁⭗⭗⧈⧈⟁', 2),  # Art of Heartflame
    ]
moon_set = [
    ('⭗⧈⟁', 1),  # Freeze Spell
    ('⭗⭗⟁', 1),  # Mechanic
    ('⭗⧈⟁⭗⭗⟁', 1),  # Art of Five Senses
    ('⟁⭗⧈⭗⟁', 3),  # Insufficient Power
    ('⟁⧈⧈⭗⧈⟁', 2),  # Tranquility Spell
    ('⭗⭗⟁⭗⧈⭗', 2),  # Stars Divine
    ]
venom_set = [
    ('⧈⧈⟁⭗⟁', 1),  # Poison Proof Spell
    ('⭗⟁⧈⟁⭗', 2),  # Anti-Toxin Spell
    ]
harmony_set = [
    ('⟁⧈⭗⟁⧈', 1),  # Art of Harmony
    ('⭗⧈⟁⭗⟁⟁', 1),  # Drunken Immortal
    ('⭗⟁⟁', 1),  # Art of Drunkness
    ('⭗⟁⧈⭗⟁⧈', 1),  # Yi Jin Jing
    ('⭗⭗⟁⧈', 1),  # Lotus in Heart
    ]
misc_set = [
    "⟁⧈⭗⧈⟁",  # Bionic Pentaform
    "⭗⧈⟁⧈⭗⧈",  # Dragonform
    '⭗⟁⧈',  # Breathing Skill
    '⧈⭗⧈⟁⭗⟁',  # Golden Acupuncture
    '⟁⟁⧈⭗⟁',  # Art of Hellish Breathing
    '⭗⟁⟁⧈⧈',  # Art of Shifting
    '⧈⧈⟁⟁',  # Turtle Preservation
    '⧈⟁⭗⭗⟁⧈',  # Traceless Heart
    '⟁⧈⟁⧈⟁',  # Blood Circulation
    ]

# To reduce the number of signs, we reduce the elemental arts down while retaining the maximum elemental chi bonus
# If you do want to over-fill an elemental chi, use the total_set function instead
mini_sun = minimal_set(sun_set)
mini_moon = minimal_set(moon_set)
mini_venom = minimal_set(venom_set)
mini_harmony = minimal_set(harmony_set)

# --- End User-Customizable Block ---

all_arts = misc_set + mini_sun + mini_moon + mini_venom + mini_harmony

work_set, graph = prepare_annealing(all_arts)
work_tree, _ = multi_anneal(work_set, graph, 0, 10)
assembled = assemble(work_tree)
totals_string, sign_totals, dip_totals, dip_sign_totals, dip_max_index = get_cost_info(assembled)

meridians = len(assembled)
signs = len(totals_string)
dip_amount = len(dip_totals)

print("Meridians: %d" % len(assembled))
print("Signs: %d (%s, %s)" % (signs, get_boosts_needed(signs), ", ".join("%dx%s" % (sign_totals[sign], sign) for sign in sign_totals)))
print("Dipping Bonuses: %d/%d (%s)" % (dip_amount, 5*signs, ", ".join("%dx%s" % (dip_sign_totals[sign], sign) for sign in dip_sign_totals)))

for i, line in enumerate(assembled):
    print(("*" if i == dip_max_index else "") + line)
