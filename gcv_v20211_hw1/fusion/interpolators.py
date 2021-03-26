import itertools
from functools import partial
from typing import List, Mapping, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy import interpolate
from tqdm import tqdm

from gcv_v20211_hw1.utils.camera_utils.camera_pose import CameraPose
from gcv_v20211_hw1.utils.camera_utils.imaging import RaycastingImaging


def get_view(
        images: List[np.array],
        distances: List[np.array],
        extrinsics: List[np.array],
        intrinsics_dict: List[Mapping],
        i):
    """A helper function to conveniently prepare view information."""
    image_i = images[i]  # [h, w]
    distances_image_i = distances[i]  # [h, w]
    # Kill background for nicer visuals
    distances_i = np.zeros_like(distances_image_i)
    distances_i[np.nonzero(image_i)] = distances_image_i[np.nonzero(image_i)]

    # TODO: write your code to constrict a world-frame point cloud from a depth image,
    #  using known intrinsic and extrinsic camera parameters.
    #  Hints: use the class `RaycastingImaging` to transform image to  points in camera frame,
    #  use the class `CameraPose` to transform image to points in world frame.

    pose_i = CameraPose(extrinsics[i])
    imaging_i = RaycastingImaging(intrinsics_dict[i]['resolution_image'], intrinsics_dict[i]['resolution_3d'])
    points_i = pose_i.camera_to_world(imaging_i.image_to_points(image_i))

    return image_i, distances_i, points_i, pose_i, imaging_i


def interpolate_ground_truth(
        images: List[np.array],
        distances: List[np.array],
        extrinsics: List[np.array],
        intrinsics_dict: List[Mapping],
):
    # Partially specify view extraction parameters.
    get_view_local = partial(get_view, images, distances, extrinsics, intrinsics_dict)

    fused_points_gt = []
    fused_predictions_gt = []
    for view_index in range(len(images)):
        image_i, distances_i, points_i, pose_i, imaging_i = get_view_local(view_index)
        fused_points_gt.append(points_i)
        fused_predictions_gt.append(distances_i.ravel()[np.flatnonzero(image_i)])

    fused_points_gt = np.concatenate(fused_points_gt)
    fused_predictions_gt = np.concatenate(fused_predictions_gt)

    return fused_points_gt, fused_predictions_gt


def pairwise_interpolate_predictions(
        view_i,
        view_j,
        indexes_j,
        distance_interpolation_threshold: float = 1.0,
        nn_set_size: int = 4,
        method='bilin',
):
    # Extract view information from input variables
    image_i, distances_i, points_i, pose_i, imaging_i = view_i
    _, distances_j, points_j, _, _ = view_j

    # Reproject points from view_j to view_i, to be able to interpolate in view_i.
    # We are using parallel projection so this explicitly computes
    # (u, v) coordinates for reprojected points (in image plane of view_i).
    # TODO: your code here: use functions from CameraPose class
    #  to transform `points_j` into coordinate frame of `view_i`
    #pose = CameraPose(pose_i)
    reprojected_j = pose_i.world_to_camera(points_j)
    # print(reprojected_j.shape)

    # For each reprojected point, find K nearest points in view_i,
    # that are source points/pixels to interpolate from.
    # We do this using imaging_i.rays_origins because these
    # define (u, v) coordinates of points_i in the pixel grid of view_i.
    # TODO: your code here: use cKDTree to find k=`nn_set_size` indexes of
    #  nearest points for each of points from `reprojected_j`
    uv_i = imaging_i.rays_origins[:, :2]
    _, nn_indexes_in_i = cKDTree(uv_i).query(reprojected_j[:, :2], k=nn_set_size)

    # Create interpolation mask: True for points which
    # can be stably interpolated (i.e. they have K neighbours present
    # within a predefined radius).
    interp_mask = np.zeros(len(reprojected_j)).astype(bool)
    # Distances to be produces as output.
    distances_j_interp = np.zeros(len(points_j), dtype=float)

    for idx, point_from_j in tqdm(enumerate(reprojected_j)):
        point_nn_indexes = nn_indexes_in_i[idx]
        # Build an [n, 3] array of XYZ coordinates for each reprojected point by taking
        # UV values from pixel grid and Z value from depth image.
        # TODO: your code here: use `point_nn_indexes` found previously
        #  and distance values from `image_i` indexed by the same `point_nn_indexes`
        point_from_j_nns = np.concatenate([uv_i[point_nn_indexes], image_i.reshape(-1)[point_nn_indexes].reshape(-1, 1)], axis=1)

        # TODO: compute a flag indicating the possibility to interpolate
        #  by checking distance between `point_from_j` and its `point_from_j_nns`
        #  against the value of `distance_interpolation_threshold`
        distances_to_nearest = np.linalg.norm(point_from_j[None, :] - point_from_j_nns, ord=2, axis=1)
        interp_mask[idx] = np.all(distances_to_nearest < distance_interpolation_threshold)

        if interp_mask[idx]:
            # Actually perform interpolation
            try:
                # TODO: your code here: use `interpolate.interp2d`
                #  to construct a bilinear interpolator from distances predicted
                #  in `view_i` (i.e. `distances_i`) into the point in `view_j`.
                #  Use the interpolator to compute an interpolated distance value.
                if method == 'bilin':
                    interpolator = interpolate.interp2d(*uv_i[point_nn_indexes].T, distances_i.reshape(-1)[point_nn_indexes])
                    distances_j_interp[idx] = interpolator(*point_from_j[:2])
                elif method == 'bispline':
                    tck = interpolate.bisplrep(*uv_i[point_nn_indexes].T, distances_i.reshape(-1)[point_nn_indexes], kx=1, ky=1)
                    distances_j_interp[idx] = interpolate.bisplev(*point_from_j[:2], tck)

            except ValueError as e:
                print('Error while interpolating point {idx}:'
                      '{what}, skipping this point'.format(
                    idx=idx, what=str(e)))
                interp_mask[idx] = False

    points_interp = points_j[interp_mask]
    indexes_interp = indexes_j[interp_mask]
    predictions_interp = distances_j_interp[interp_mask]

    return predictions_interp, indexes_interp, points_interp


def multi_view_interpolate_predictions(
        images: List[np.array],
        distances: List[np.array],
        extrinsics: List[np.array],
        intrinsics_dict: List[Mapping],
        **interpolation_params,
) -> Tuple[List, List, List]:
    """Interpolated predictions between views.

    :param images: list of 2d depth images
    :param distances: list of 2d distance-to-feature predictions
    :param extrinsics: list of 4x4 camera extrinsic (camera->world) matrices
    :param intrinsics_dict: list of imaging parameters for parallel projection
    :param interpolation_params: parameters for interpolation procedure

    :return: tuple of interpolated predictions, indexes, and points
    """
    # Partially specify view extraction parameters.
    get_view_local = partial(get_view, images, distances, extrinsics, intrinsics_dict)

    # Prepare output arrays
    list_predictions = []  # list of 1-d predictions (List[array.shape==[n, ])
    list_indexes_in_whole = []  # list of indexes into global set of points (List[array.shape==[n, ])
    list_points = []  # list of 3-d points (List[array.shape==[n, 3])

    # 0 to n-1 indexes into global set of points for an object
    point_indexes = np.cumsum([len(np.flatnonzero(image)) for image in images])

    # Iterate over each pair of depth images, trying to interpolate
    # from view i into view j
    n_images = len(images)
    print(f'NUMBER OF IMAGES: {len(images)}')

    for i, j in itertools.product(range(n_images), range(n_images)):
        print(f'Pair {i} {j}')
        # Extract view information: view_i is a tuple
        view_i, view_j = get_view_local(i), get_view_local(j)

        # Construct list of indexes (for currently processed points_j)
        # into global set of points for view_i
        start_idx, end_idx = (0, point_indexes[j]) if 0 == j \
            else (point_indexes[j - 1], point_indexes[j])
        indexes_in_whole = np.arange(start_idx, end_idx)

        if i == j:
            # Simply add predictions from view_i into the result
            image_i, distances_i, points_i, pose_i, imaging_i = view_i
            predictions_interp, indexes_interp, points_interp = \
                distances_i[image_i != 0.].ravel(), indexes_in_whole, points_i

        else:
            # Actually run interpolation to label points in view_j
            # with predictions obtained by interpolating from view_i
            predictions_interp, indexes_interp, points_interp = pairwise_interpolate_predictions(
                view_i,
                view_j,
                indexes_in_whole,
                **interpolation_params)

        list_predictions.append(predictions_interp)
        list_indexes_in_whole.append(indexes_interp)
        list_points.append(points_interp)

    return list_predictions, list_indexes_in_whole, list_points
