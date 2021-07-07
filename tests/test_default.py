import pytest
import compas
import gkr_mult_drt_robot.solver.multiple_directions_calculator

if __name__ == '__main__':
    test_graph =[[1, 2, 0.519615242271, -0.3, -0.8],
                 [1, 2, 0.866025403784, -0.5, -6.12323399574e-17],
                 [1, 3, 0.519615242271, 0.3, -0.8],
                 [1, 3, 0.866025403784, 0.5, 6.12323399574e-17],
                 [2, 3, 0.0257610515846, 0.999035371738, -0.0355625397974],
                 ]

    candidate_parts = [2, 3]
    install_parts = [0, 1]
    target_directions = [[0, 0, 1], [0, 0, 1]]

    result = gkr_mult_drt_robot.solver.multiple_directions_calculator.compute_simultaneous_translational_disassembling_directions_without_guidance(test_graph, candidate_parts, install_parts)
    print(result)

    result = gkr_mult_drt_robot.solver.multiple_directions_calculator.compute_simultaneous_translational_disassembling_directions_with_guidance(test_graph, candidate_parts, install_parts, target_directions)
    print(result)

