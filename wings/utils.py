import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def order_coords(predicted, original):
    predicted = np.array(predicted)
    original = np.array(original)

    original = np.array(list(zip(original[::2], original[1::2])))
    distance_matrix = cdist(original, predicted)
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    # Does not preserve the original coords order
    matched_predicted = predicted[col_ind]
    matched_original = original[row_ind]

    # Preserves the original coords order
    # matched_predicted = np.zeros_like(original)
    # for orig_idx, pred_idx in zip(row_ind, col_ind):
    #     matched_predicted[orig_idx] = predicted[pred_idx]

    return matched_predicted, matched_original
