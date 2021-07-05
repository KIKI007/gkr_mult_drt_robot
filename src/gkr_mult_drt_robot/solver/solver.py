from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import cvxpy as cp

def compute_simultaneous_translational_disassembling_directions(graph, candidate_parts, installed_parts):

    map_candidate = {}
    map_boundary = {}
    for id in range(0, len(candidate_parts)):
        partID = candidate_parts[id]
        map_candidate[partID] = id

    for part in installed_parts:
        map_boundary[part] = True

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


    v = cp.Variable(len(candidate_parts) * 3)
    t = cp.Variable(len(trimmed_graph))

    constraints = []
    for id in range(0, len(trimmed_graph)):

        data = trimmed_graph[id]
        partI = data[0]
        partJ = data[1]
        normal = np.array([data[2], data[3], data[4]])

        # both are candiadate part
        delta_v = 0
        if map_candidate.get(partI) != None and map_candidate.get(partJ) != None:
            vI = map_candidate[partI]
            vJ = map_candidate[partJ]
            delta_v = v[vJ * 3: vJ * 3 + 3] - v[vI * 3: vI * 3 + 3]
            #print("case 1:", partI, partJ)
            #print(partI, partJ, vI, vJ)
        elif map_candidate.get(partI) == None and map_candidate.get(partJ) != None:
            vJ = map_candidate[partJ]
            delta_v = v[vJ * 3: vJ * 3 + 3]
            #print(partI, partJ, vJ)
            #print("case 2:", partI, partJ)
        elif map_candidate.get(partI) != None and map_candidate.get(partJ) == None:
            vI = map_candidate[partI]
            delta_v = - v[vI * 3: vI * 3 + 3]
            #print(partI, partJ, vI)
            #print("case 3:", partI, partJ)
        constraints.append(normal.T@delta_v >= t[id])

    constraints.append(-1 <= v)
    constraints.append(v <= 1)
    constraints.append(t >= 0)
    objective = cp.Maximize(cp.sum(t))
    prob = cp.Problem(objective,constraints)
    prob.solve()

    if prob.value < 1E-4:
        return [False, []]
    else:
        velocities = []
        v_value = v.value
        for id in range(0, len(candidate_parts)):
            velocities.append([v_value[id * 3], v_value[id * 3 + 1], v_value[id * 3 + 2]])
        return [True, velocities]