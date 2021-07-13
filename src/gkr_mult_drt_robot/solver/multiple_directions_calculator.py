from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import cvxpy as cp
from gkr_mult_drt_robot.solver.seperating_axis import compute_optimal_separating_plane

def append_graph_additional_contact_avoiding_nonneighbouring_collision(graph, candidate_parts, candidate_parts_pts):
    part_pairs = []

    for id in range(0, len(candidate_parts)):
        for jd in range(id + 1, len(candidate_parts)):
            part_pairs.append([candidate_parts[id], candidate_parts[jd]])

    # remove part pairs that are in the neighbour
    for data in graph:
        if [data[0], data[1]] in part_pairs:
            part_pairs.remove([data[0], data[1]])

    to_local_index = {}
    index = 0
    for partID in candidate_parts:
        to_local_index[partID] = index
        index = index + 1

    for part_pair in part_pairs:
        part0 = to_local_index[part_pair[0]]
        part1 = to_local_index[part_pair[1]]
        result = compute_optimal_separating_plane(candidate_parts_pts[part0], candidate_parts_pts[part1])
        if result != None:
            graph.append([part_pair[0], part_pair[1], *result[0][0]])

def trim_unrelated_graph_edges(graph, map_candidate, map_boundary):

    # remove unrelated graph edge
    trimmed_graph = []
    for data in graph:
        partI = data[0]
        partJ = data[1]

        # if part is not candidate part and not installed part, skip
        if map_candidate.get(partI) == None and map_boundary.get(partI) == None:
            continue
        if map_candidate.get(partJ) == None and map_boundary.get(partJ) == None:
            continue

        # if either part is not candidate part, skip
        if map_candidate.get(partI) == None and map_candidate.get(partJ) == None:
            continue

        trimmed_graph.append(data)

    return trimmed_graph

def convert_graph_into_dictionary_maps(graph, candidate_parts, installed_parts):
    map_candidate = {}
    map_boundary = {}
    for id in range(0, len(candidate_parts)):
        partID = candidate_parts[id]
        map_candidate[partID] = id

    for part in installed_parts:
        map_boundary[part] = True

    return [map_candidate, map_boundary]

def obtain_optimization_constraints_from_contact_graph(graph, map_candidate, num_of_parts, v, t, with_guidance = False):
    constraints = []
    for id in range(0, len(graph)):

        data = graph[id]
        partI = data[0]
        partJ = data[1]
        normal = np.array([data[2], data[3], data[4]])

        # both are candiadate part
        delta_v = 0
        if map_candidate.get(partI) != None and map_candidate.get(partJ) != None:
            vI = map_candidate[partI]
            vJ = map_candidate[partJ]
            delta_v = v[vJ * 3: vJ * 3 + 3] - v[vI * 3: vI * 3 + 3]
        elif map_candidate.get(partI) == None and map_candidate.get(partJ) != None:
            vJ = map_candidate[partJ]
            delta_v = v[vJ * 3: vJ * 3 + 3]
        elif map_candidate.get(partI) != None and map_candidate.get(partJ) == None:
            vI = map_candidate[partI]
            delta_v = - v[vI * 3: vI * 3 + 3]

        if with_guidance:
            constraints.append(normal.T @ delta_v >= 0)
        else:
            constraints.append(normal.T @ delta_v >= t[id])

    for id in range(0, num_of_parts):
        constraints.append(cp.SOC(1, v[id * 3: id * 3 + 3]))

    return constraints

def obtain_optimization_constraints_from_contact_graph_with_guidance(graph, map_candidate, num_of_parts, v):
    t = cp.Variable(1)
    return obtain_optimization_constraints_from_contact_graph(graph, map_candidate, num_of_parts, v, t, True)


def obtain_optimization_result(prob, candidate_parts, v):
    if prob.value < 1E-4:
        return [False, []]
    else:
        velocities = []
        v_value = v.value
        for id in range(0, len(candidate_parts)):
            velocities.append([v_value[id * 3], v_value[id * 3 + 1], v_value[id * 3 + 2]])
        return [True, velocities]

def compute_simultaneous_translational_disassembling_directions_avoiding_nonneighbor_collision(graph,
                                                                                               candidate_parts,
                                                                                               installed_parts,
                                                                                               candidate_parts_pts):

    [map_candidate, map_boundary] = convert_graph_into_dictionary_maps(graph, candidate_parts, installed_parts)
    trimmed_graph = trim_unrelated_graph_edges(graph, map_candidate, map_boundary)
    append_graph_additional_contact_avoiding_nonneighbouring_collision(trimmed_graph, candidate_parts, candidate_parts_pts)

    num_of_parts = len(candidate_parts)
    v = cp.Variable(num_of_parts * 3)
    t = cp.Variable(len(trimmed_graph))

    constraints = obtain_optimization_constraints_from_contact_graph(trimmed_graph, map_candidate, num_of_parts, v, t, False)
    constraints.append(t >= 0)

    objective = cp.Maximize(cp.sum(t))
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return obtain_optimization_result(prob, candidate_parts, v)

def compute_simultaneous_translational_disassembling_directions_without_guidance(graph,
                                                                                 candidate_parts,
                                                                                 installed_parts):

    [map_candidate, map_boundary] = convert_graph_into_dictionary_maps(graph, candidate_parts, installed_parts)
    trimmed_graph = trim_unrelated_graph_edges(graph, map_candidate, map_boundary)

    num_of_parts = len(candidate_parts)
    v = cp.Variable(num_of_parts * 3)
    t = cp.Variable(len(trimmed_graph))

    constraints = obtain_optimization_constraints_from_contact_graph(trimmed_graph, map_candidate, num_of_parts, v, t, False)
    constraints.append(t >= 0)

    objective = cp.Maximize(cp.sum(t))
    prob = cp.Problem(objective,constraints)
    prob.solve()

    return obtain_optimization_result(prob, candidate_parts, v)

def compute_simultaneous_translational_disassembling_directions_with_guidance(graph, candidate_parts, installed_parts, target_directions):

    [map_candidate, map_boundary] = convert_graph_into_dictionary_maps(graph, candidate_parts, installed_parts)
    trimmed_graph = trim_unrelated_graph_edges(graph, map_candidate, map_boundary)

    num_of_parts = len(candidate_parts)
    v = cp.Variable(num_of_parts * 3)

    constraints = obtain_optimization_constraints_from_contact_graph_with_guidance(trimmed_graph, map_candidate, num_of_parts, v)

    objective_func = 0
    for id in range(0, num_of_parts):
        drt = np.array(target_directions[id])
        objective_func += drt.T @ v[id * 3: id * 3 + 3]
    objective = cp.Maximize(objective_func)
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return obtain_optimization_result(prob, candidate_parts, v)
