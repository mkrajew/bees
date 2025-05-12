import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# # Your original and predicted lists
# original = np.array([...])  # shape (19, 2)
# predicted = np.array([...])  # shape (19, 2)
#
# # Compute pairwise Euclidean distances
# distance_matrix = cdist(original, predicted)
#
# # Solve the assignment problem
# row_ind, col_ind = linear_sum_assignment(distance_matrix)
#
# # Get matched pairs
# matched_original = original[row_ind]
# matched_predicted = predicted[col_ind]
#
# # Optionally, view the pairs and distances
# for i, j in zip(row_ind, col_ind):
#     print(f"Original: {original[i]}, Predicted: {predicted[j]}, Distance: {distance_matrix[i, j]:.2f}")
#
#


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