import numpy as np

from gcv_v20211_hw1.utils.camera_utils.raycasting import generate_rays


class RaycastingImaging:
    def __init__(self, resolution_image, resolution_3d):
        self.resolution_image = resolution_image
        self.resolution_3d = resolution_3d
        self.rays_screen_coords, self.rays_origins, self.rays_directions = generate_rays(
            self.resolution_image, self.resolution_3d)

    def points_to_image(self, points, ray_indexes, assign_channels=None):
        xy_to_ij = self.rays_screen_coords[ray_indexes]
        # note that `points` can have many data dimensions
        if None is assign_channels:
            assign_channels = [2]
        data_channels = len(assign_channels)
        image = np.zeros((self.resolution_image, self.resolution_image, data_channels))
        # rays origins (h, w, 3), z is the same for all points of matrix
        # distance is absolute value
        image[xy_to_ij[:, 0], xy_to_ij[:, 1]] = points[:, assign_channels]
        return image.squeeze()

    def image_to_points(self, image):
        i = np.where(image.ravel() != 0)[0]
        points = np.zeros((len(i), 3))
        points[:, 0] = self.rays_origins[i, 0]
        points[:, 1] = self.rays_origins[i, 1]
        points[:, 2] = image.ravel()[i]
        return points
