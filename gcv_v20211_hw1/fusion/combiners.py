from collections import defaultdict
from typing import List, Mapping, Tuple

import numpy as np
from tqdm import tqdm


def combine_predictions(
        n_points: int,
        list_predictions: List[np.array],
        list_indexes_in_whole: List[np.array],
        list_points: List[np.array],
) -> Tuple[np.array, Mapping]:

    """Given a point cloud with more than one distance-to-feature
    prediction per each of the 3D points, compute a single
    distance-to-feature prediction per each of the 3D points.

    :param n_points: total number of points in a point cloud
    :param list_predictions: list of numpy arrays corresponding to
        predictions in each 3D point
    :param list_indexes_in_whole:
    :param list_points:
    :return: a list of predictions
    """
    fused_predictions = np.ones(n_points) * np.inf

    # step 1: gather predictions
    predictions_variants = defaultdict(list)
    iterable = zip(list_predictions, list_indexes_in_whole, list_points)
    for distances, indexes_gt, points_gt in tqdm(iterable):
        for i, idx in enumerate(indexes_gt):
            predictions_variants[idx].append(distances[i])

    # step 2: consolidate predictions
    for idx, values in predictions_variants.items():
        fused_predictions[idx] = np.min(values)

    return fused_predictions, predictions_variants
