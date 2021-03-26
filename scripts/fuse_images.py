#!/usr/bin/env python3

# Usage:
#   python fuse_images.py -t input_gt.hdf5 -p input_pred.hdf5 -o output_dir/

import argparse
import os
import sys

import numpy as np

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..'))
sys.path[1:1] = [__dir__]

import gcv_v20211_hw1.utils.sharpf_io as sharpf_io
from gcv_v20211_hw1.utils.hdf5.dataset import Hdf5File, PreloadTypes
from gcv_v20211_hw1.fusion.combiners import combine_predictions
import gcv_v20211_hw1.fusion.interpolators as interpolators

HIGH_RES = 0.02
MED_RES = 0.05
LOW_RES = 0.125


def main(options):
    # extract a filename from the input pathname to use further
    name = os.path.splitext(os.path.basename(options.true_filename))[0]

    # load ground truth images and distances
    print('Loading ground truth data...')
    gt_dataset = Hdf5File(
        options.true_filename,
        io=sharpf_io.WholeDepthMapIO,
        preload=PreloadTypes.LAZY,
        labels='*')
    # depth images captured from a variety of views around the 3D shape
    gt_images = [view['image'] for view in gt_dataset]
    # ground-truth distances (multi-view consistent for the global 3D shape)
    gt_distances = [view.get('distances', np.ones_like(view['image'])) for view in gt_dataset]
    # extrinsic camera matrixes describing the 3D camera poses used to capture depth images
    gt_extrinsics = [view['camera_pose'] for view in gt_dataset]
    # intrinsic camera parameters describing how to compute image from points and vice versa
    gt_intrinsics = [dict(resolution_image=gt_images[0].shape, resolution_3d=options.resolution_3d) for view in gt_dataset]

    # construct the globally consistent 3D point cloud
    # from a list of individual image-distances pairs
    print('Fusing ground truth data...')
    fused_points_gt, \
    fused_distances_gt = interpolators.interpolate_ground_truth(
        gt_images,
        gt_distances,
        gt_extrinsics,
        gt_intrinsics)
    n_points = len(fused_points_gt)

    # save point cloud with ground-truth distance-to-feature values to an output file
    gt_output_filename = os.path.join(
        options.output_dir,
        '{}__{}.hdf5'.format(name, 'ground_truth'))
    print('Saving ground truth to {}'.format(gt_output_filename))
    sharpf_io.save_full_model_predictions(
        fused_points_gt,
        fused_distances_gt,
        gt_output_filename)

    # load predicted distances
    print('Loading predictions...')
    predictions_dataset = Hdf5File(
        options.pred_filename,
        io=sharpf_io.WholeDepthMapIO,
        preload=PreloadTypes.LAZY,
        labels='*')
    # predicted distances (NOT multi-view consistent, as each CNN only had access to a particular view)
    pred_distances = [view['distances'] for view in predictions_dataset]

    # Interpolate predictions from individual views by re-projecting them
    # from view i to view j for each pair (i, j) of views,
    # obtaining several predictions for each measured point.
    print('Interpolating predictions...')
    threshold = options.resolution_3d * options.distance_interp_factor
    list_predictions, \
    list_indexes_in_whole, \
    list_points = interpolators.multi_view_interpolate_predictions(
        gt_images,
        pred_distances,
        gt_extrinsics,
        gt_intrinsics,
        nn_set_size=options.nn_set_size,
        distance_interpolation_threshold=threshold)

    # Now that we have obtained a set of predictions per each individual point,
    # we can combine distance-to-feature predictions into a consolidated
    # point cloud by simply taking min value across all distance-to-feature predictions
    print('Fusing predictions...')
    combined_predictions, \
    prediction_variants = \
        combine_predictions(
            n_points,
            list_predictions,
            list_indexes_in_whole,
            list_points)

    # save point cloud with predicted distance-to-feature values to an output file
    pred_output_filename = os.path.join(
        options.output_dir,
        '{}__{}.hdf5'.format(name, 'interpolated'))
    print('Saving predictions to {}'.format(pred_output_filename))
    sharpf_io.save_full_model_predictions(
        fused_points_gt,
        combined_predictions,
        pred_output_filename)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--true-filename', dest='true_filename', required=True,
                        help='Path to GT file with whole model point patches.')
    parser.add_argument('-p', '--pred-filename', dest='pred_filename', required=True,
                        help='Path to prediction directory with npy files.')
    parser.add_argument('-o', '--output-dir', dest='output_dir', required=True,
                        help='Path to output (suffixes indicating various methods will be added).')
    parser.add_argument('-k', '--nn_set_size', dest='nn_set_size', required=False, default=4, type=int,
                        help='Number of neighbors used for interpolation.')
    parser.add_argument('-r', '--resolution_3d', dest='resolution_3d', required=False, default=HIGH_RES, type=float,
                        help='3D resolution of scans.')
    parser.add_argument('-f', '--distance_interp_factor', dest='distance_interp_factor', required=False, type=float, default=6.,
                        help='distance_interp_factor * resolution_3d is the distance_interpolation_threshold')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
