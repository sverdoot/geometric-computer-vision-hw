from collections import defaultdict
from typing import List, Mapping, Tuple

import numpy as np
from tqdm import tqdm


def combine_predictions(
        n_points: int,
        list_predictions: List[np.array],
        list_indexes_in_whole: List[np.array],
        list_points: List[np.array],
        aggregation_method='min',
        postprocessing=None,
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
        if aggregation_method == 'min':
            fused_predictions[idx] = np.min(values)
        elif aggregation_method == 'truncated_min':
            # Truncated average/min, computed by removing the
            # largest and smallest 20% of values, then computing the
            # corresponding quantity.
            values = np.sort(values)
            values = values[int(0.2*len(values)):int(0.8*len(values))]
            fused_predictions[idx] = np.min(values)
        elif aggregation_method == 'truncated_median':
            values = np.sort(values)
            values = values[int(0.2*len(values)):int(0.8*len(values))]
            fused_predictions[idx] = np.median(values)
        elif aggregation_method == 'truncated_mean':
            values = np.sort(values)
            values = values[int(0.2*len(values)):int(0.8*len(values))]
            fused_predictions[idx] = np.mean(values)

        # if postprocessing is not None:
        #     if postprocessing == 'L2':
                

    return fused_predictions, predictions_variants
