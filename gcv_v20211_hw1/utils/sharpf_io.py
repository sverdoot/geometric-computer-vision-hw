from functools import partial

import h5py
import numpy as np

import gcv_v20211_hw1.utils.hdf5.io_struct as io


WholeDepthMapIO = io.HDF5IO({
        'image': io.Float64('image'),
        'normals': io.Float64('normals'),
        'distances': io.Float64('distances'),
        'directions': io.Float64('directions'),
        'indexes_in_whole': io.Int32('indexes_in_whole'),
        'item_id': io.AsciiString('item_id'),
        'orig_vert_indices': io.VarInt32('orig_vert_indices'),
        'orig_face_indexes': io.VarInt32('orig_face_indexes'),
        'has_sharp': io.Bool('has_sharp'),
        'num_sharp_curves': io.Int8('num_sharp_curves'),
        'num_surfaces': io.Int8('num_surfaces'),
        'camera_pose': io.Float64('camera_pose'),
        'mesh_scale': io.Float64('mesh_scale'),
        'has_smell_coarse_surfaces_by_num_faces': io.Bool('has_smell_coarse_surfaces_by_num_faces'),
        'has_smell_coarse_surfaces_by_angles': io.Bool('has_smell_coarse_surfaces_by_angles'),
        'has_smell_deviating_resolution': io.Bool('has_smell_deviating_resolution'),
        'has_smell_sharpness_discontinuities': io.Bool('has_smell_sharpness_discontinuities'),
        'has_smell_bad_face_sampling': io.Bool('has_smell_bad_face_sampling'),
        'has_smell_mismatching_surface_annotation': io.Bool('has_smell_mismatching_surface_annotation'),
        'has_smell_raycasting_background': io.Bool('has_smell_raycasting_background'),
        'has_smell_depth_discontinuity': io.Bool('has_smell_depth_discontinuity'),
        'has_smell_mesh_self_intersections': io.Bool('has_smell_mesh_self_intersections'),
    },
    len_label='has_sharp',
    compression='lzf')


def save_whole_images(patches, filename):
    # turn a list of dicts into a dict of torch tensors:
    # default_collate([{'a': 'str1', 'x': np.random.normal()}, {'a': 'str2', 'x': np.random.normal()}])
    # Out[26]: {'a': ['str1', 'str2'], 'x': tensor([0.4252, 0.1414], dtype=torch.float64)}
    collate_fn = partial(io.collate_mapping_with_io, io=WholeDepthMapIO)
    patches = collate_fn(patches)

    with h5py.File(filename, 'w') as f:
        for key in ['image', 'normals', 'distances', 'directions', 'indexes_in_whole']:
            WholeDepthMapIO.write(f, key, patches[key].numpy())
        WholeDepthMapIO.write(f, 'item_id', patches['item_id'])
        WholeDepthMapIO.write(f, 'orig_vert_indices', patches['orig_vert_indices'])
        WholeDepthMapIO.write(f, 'orig_face_indexes', patches['orig_face_indexes'])
        WholeDepthMapIO.write(f, 'has_sharp', patches['has_sharp'].numpy().astype(np.bool))
        WholeDepthMapIO.write(f, 'num_sharp_curves', patches['num_sharp_curves'].numpy())
        WholeDepthMapIO.write(f, 'num_surfaces', patches['num_surfaces'].numpy())
        WholeDepthMapIO.write(f, 'camera_pose', patches['camera_pose'].numpy())
        WholeDepthMapIO.write(f, 'mesh_scale', patches['mesh_scale'].numpy())
        has_smell_keys = [key for key in WholeDepthMapIO.datasets.keys()
                          if key.startswith('has_smell')]
        for key in has_smell_keys:
            WholeDepthMapIO.write(f, key, patches[key].numpy().astype(np.bool))


PointPatchPredictionsIO = io.HDF5IO({
    'points': io.Float64('points'),
    'distances': io.Float64('distances')
},
    len_label='distances',
    compression='lzf')


def save_full_model_predictions(points, predictions, filename):
    with h5py.File(filename, 'w') as f:
        PointPatchPredictionsIO.write(f, 'points', [points])
        PointPatchPredictionsIO.write(f, 'distances', [predictions])
