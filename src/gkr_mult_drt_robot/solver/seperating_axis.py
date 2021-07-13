from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
from scipy.spatial import ConvexHull

axis_directions_dot_product_eps = 1e-3
axis_direction_norm_eps = 1e-4

def compute_candidate_separating_axes_from_part_faces(part_convexhull):
    directions = []
    for s in part_convexhull.simplices:

        # Each simplex in part_convexhull is a 3D triangle
        pA = part_convexhull.points[s[0]]
        pB = part_convexhull.points[s[1]]
        pC = part_convexhull.points[s[2]]

        # Compute the normal of the triangle face
        normal = np.cross(pB - pA, pC - pA)
        if np.linalg.norm(normal) > axis_direction_norm_eps:
            normal /= np.linalg.norm(normal)
            directions.append(normal)

    return directions

def remove_duplicate_candidate_separating_axes(directions):
    trimmed_directions = []
    for direction in directions:
        is_duplicate = False
        for trimmed_direction in trimmed_directions:
            dot_product = np.dot(trimmed_direction, direction)

            # -v, v both will be considered as duplicated vector
            if abs(1.0 - dot_product) < axis_directions_dot_product_eps\
                or abs(1.0 + dot_product) < axis_directions_dot_product_eps:
                is_duplicate = True
                break

        if is_duplicate == False:
            trimmed_directions.append(direction)

    return trimmed_directions

def compute_separating_plane(pts0, pts1, axis):
    # check [axis, -axis]
    # the axis direction should always keep the part1 at the positive side of the separating plane
    # and keep the part0 at the negative side of the separating plane.

    for drt in [axis, -axis]:

        distance0 = [np.dot(pt, drt) for pt in pts0]
        distance1 = [np.dot(pt, drt) for pt in pts1]

        min_distance1 = min(distance1)
        max_distance0 = max(distance0)

        if min_distance1 > max_distance0:
            D = (min_distance1 + max_distance0) / 2
            return [drt, D, min_distance1 - max_distance0]

    return [None, None, -1]

def compute_candidate_separating_axes(hull0, hull1):
    axes0 = compute_candidate_separating_axes_from_part_faces(hull0)
    axes1 = compute_candidate_separating_axes_from_part_faces(hull1)
    axes = remove_duplicate_candidate_separating_axes([*axes0, *axes1])
    return axes

def compute_optimal_separating_plane(block_pts0, block_pts1):

    hull0 = ConvexHull(block_pts0)
    hull1 = ConvexHull(block_pts1)
    vertices0 = hull0.points[hull0.vertices]
    vertices1 = hull1.points[hull1.vertices]

    axes = compute_candidate_separating_axes(hull0, hull1)

    max_distance = -1E6
    result_plane = []
    for axis in axes:
        [drt, D, distance] = compute_separating_plane(vertices0, vertices1, axis)
        if distance > max_distance and D != None:
            max_distance = distance
            result_plane = [drt, D]

    if max_distance > 1E-6:
        return [result_plane, max_distance]
    else:
        return None
