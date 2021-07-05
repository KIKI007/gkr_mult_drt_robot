import pytest
import compas
import gkr_mult_drt_robot.solver


if __name__ == '__main__':
    test_graph = [[0, 1, 0.430052637505, -0.874602109355, -0.223888095457],
                  [0, 3, -0.74386922067, 0.451003127383, 0.493208639047],
                  [1, 2, -0.3368191474, -0.923253177958, 0.184814586371],
                  [2, 4, -0.730619816405, -0.428084349835, 0.531919611692],
                  [3, 4, -0.0257610515846, -0.999035371738, 0.0355625397974]]

    candidate_parts = [1, 2, 3]
    install_parts = [0]

    result = gkr_mult_drt_robot.solver.compute_simultaneous_translational_disassembling_directions(test_graph, candidate_parts, install_parts)
    print(result)
