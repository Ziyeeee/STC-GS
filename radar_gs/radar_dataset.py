import os
import sys
from pathlib import Path
from typing import NamedTuple
import random
import h5py
from plyfile import PlyData, PlyElement
from scipy.interpolate import interpn
import numpy as np
import torch


class CameraInfo(NamedTuple):
    R: np.array
    T: np.array
    view_depth: float
    gt_image: np.array
    channel: int
    width: int
    height: int


class SceneInfo:
    def __init__(self, point_cloud=None, xoy=None, supplement=None):
        self.point_cloud = point_cloud
        self.xoy = xoy
        self.supplement = supplement

    def __len__(self):
        return len(self.xoy) + len(self.supplement)

    def set_xoy(self, xoy):
        self.xoy = xoy

    def set_supplement(self, supplement):
        self.supplement = supplement


def get_xoy(np_data: np.array, view_depth: int):
    return np_data[:, view_depth, :, :]


def get_xoz(np_data: np.array, view_depth: int):
    return np_data[:, :, view_depth]


def get_yoz(np_data: np.array, view_depth: int):
    return np_data[:, view_depth, :]


def get_tilted_xoy_orthogonal_yoz(np_data: np.array, alpha: float, view_depth: int):
    C, D, H, W = np_data.shape
    abs_sin_alpha = abs(np.sin(alpha))
    cos_alpha = np.cos(alpha)
    x_idx = np.arange(0, H, 1)
    temp_idx = np.arange(0, D / abs_sin_alpha, 1)
    X, Temp = np.meshgrid(x_idx, temp_idx)

    points = (np.arange(C), np.arange(D), np.arange(H), np.arange(W))
    if alpha >= 0:
        xi = np.stack([Temp * abs_sin_alpha, X, Temp * cos_alpha + view_depth], axis=-1)
    else:
        xi = np.stack([D - Temp * abs_sin_alpha, X, Temp * cos_alpha + view_depth], axis=-1)

    xi = xi[None].repeat(C, axis=0)
    ci = np.arange(C)[:, None, None, None].repeat(xi.shape[1], axis=1).repeat(xi.shape[2], axis=2)
    xi = np.concatenate([ci, xi], axis=-1)
    return interpn(points, np_data, xi, method="linear", bounds_error=False, fill_value=0)


def get_tilted_xoy_orthogonal_xoz(np_data: np.array, alpha: float, view_depth: int):
    C, D, H, W = np_data.shape
    abs_sin_alpha = abs(np.sin(alpha))
    cos_alpha = np.cos(alpha)
    y_idx = np.arange(0, W, 1)
    temp_idx = np.arange(0, D / abs_sin_alpha, 1)
    Y, Temp = np.meshgrid(y_idx, temp_idx)

    points = (np.arange(C), np.arange(D), np.arange(H), np.arange(W))
    if alpha >= 0:
        xi = np.stack([Temp * abs_sin_alpha, Temp * cos_alpha + view_depth, Y], axis=-1)
    else:
        xi = np.stack([D - Temp * abs_sin_alpha, Temp * cos_alpha + view_depth, Y], axis=-1)

    xi = xi[None].repeat(C, axis=0)
    ci = np.arange(C)[:, None, None, None].repeat(xi.shape[1], axis=1).repeat(xi.shape[2], axis=2)
    xi = np.concatenate([ci, xi], axis=-1)
    return interpn(points, np_data, xi, method="linear", bounds_error=False, fill_value=0)


class RadarDataset:
    def __init__(self, max_init_points, num_vertical_samples, seq_len) -> None:
        self.max_init_points = max_init_points
        self.num_vertical_samples = num_vertical_samples
        self.seq_len = seq_len
        self.M = self.get_vertical_interpolation_matrix()
        self.M = torch.tensor(self.M, dtype=torch.float32)

    def preprocess(self, hdf_path, seq_name, device="cpu"):
        with h5py.File(hdf_path, "r") as f:
            group = f[seq_name]

            keys = ["Z_H", "SW", "AzShr", "Div", "Z_DR", "K_DP"]
            # Z_H: 0 - 70
            # SW: 0 - 20
            # AzShr: -0.02 - 0.02
            # Div: -0.02 - 0.02
            # Z_DR: -20 - 20
            # K_DP: -50 - 50

            data = np.stack([group[key][:] for key in keys], axis=0)
            # radar signals in the 0.5km and 1.0km are not complete
            data = data[:, 2:]
        # normalize
        data = data / np.array([70, 20, 0.02, 0.02, 20, 50]).reshape(-1, 1, 1, 1)
        data = torch.tensor(data, dtype=torch.float32, device=device)
        # interpolate to even vertical resolution
        data = torch.tensordot(self.M.to(data.device), data, dims=([1], [1]))
        data = torch.clamp(data, -1, 1).transpose(1, 0).float()
        # too high to have little information
        data = data[:, :-6]

        return data

    def generateRadarSceneInfo(self, hdf_path, seq_name, device="cpu", init_ply=False):
        data = self.preprocess(hdf_path, seq_name, device)

        if init_ply:
            ply_data = self.generateInitPoints(data.cpu().numpy())
        else:
            ply_data = None
        xoy_cam_infos, supplementary_cam_infos = self.generateRadarCameras(data.cpu().numpy())

        scene_info = SceneInfo(point_cloud=ply_data, xoy=xoy_cam_infos, supplement=supplementary_cam_infos)
        return scene_info, data[0] * 255

    def generateInitPoints(self, data):
        sample_rate = 0.1
        C, D, H, W = data.shape
        mask = data[0] > 0.01
        data[:, ~mask] = 0
        z, x, y = mask.nonzero()
        if len(x) * sample_rate >= self.max_init_points:
            indices = random.sample(range(len(x)), self.max_init_points)
            x = x[indices]
            y = y[indices]
            z = z[indices]
        else:
            indices = random.sample(range(len(x)), int(len(x) * sample_rate))
            valid_x = x[indices]
            valid_y = y[indices]
            valid_z = z[indices]
            empty_z, empty_x, empty_y = (~mask).nonzero()
            indices = random.sample(range(len(empty_x)), self.max_init_points - int(len(x) * sample_rate))
            x = np.concatenate((valid_x, empty_x[indices]))
            y = np.concatenate((valid_y, empty_y[indices]))
            z = np.concatenate((valid_z, empty_z[indices]))

        # add noise to avoid trapped in local minimum
        pos_x = x.astype(np.float32) + np.random.randn(len(x)) * 0.5
        pos_y = y.astype(np.float32) + np.random.randn(len(y)) * 0.5
        pos_z = z.astype(np.float32) + np.random.randn(len(z)) * 0.5

        dtype_full = [
            (attribute, "f4")
            for attribute in [
                "x",
                "y",
                "z",
                "Z_H",
                "SW",
                "AzShr",
                "Div",
                "Z_DR",
                "K_DP",
                "scale_0",
                "scale_1",
                "scale_2",
                "rot_0",
                "rot_1",
                "rot_2",
                "rot_3",
            ]
        ]
        # shift to pixel coordinate
        position = np.stack((pos_x - H * 0.5 + 0.5, pos_y - W * 0.5 + 0.5, pos_z), axis=1)
        features = data[:, z, x, y].transpose(1, 0)
        scale_rot = np.zeros((len(x), 7))
        scale_rot[:, 3] = 1.0
        elements = list(map(tuple, np.concatenate((position, features, scale_rot), axis=1)))
        el = PlyElement.describe(np.array(elements, dtype=dtype_full), "gaussians")
        return PlyData([el])

    def generateRadarSceneInfoWithInverseGaussian(self, hdf_path, seq_name, gaussians_tensor, energy_img, device="cpu"):
        data = self.preprocess(hdf_path, seq_name, device)

        C, D, H, W = data.shape
        N = gaussians_tensor.shape[0]

        cover_mask = (data[0] > 0.01) & (energy_img > 0.01)
        uncover_mask = (data[0] > 0.01) & (energy_img < 0.01)

        # re-sample the region that is not covered during the pre-reconstruction
        ratio = uncover_mask.sum() / (cover_mask.sum() + uncover_mask.sum())
        z, x, y = uncover_mask.cpu().numpy().nonzero()
        indices = random.sample(range(len(x)), int(N * ratio) if int(N * ratio) < len(x) else len(x))
        x = x[indices]
        y = y[indices]
        z = z[indices]

        # add noise to avoid trapped in local minimum
        pos_x = x.astype(np.float32) + np.random.randn(len(x)) * 0.5
        pos_y = y.astype(np.float32) + np.random.randn(len(y)) * 0.5
        pos_z = z.astype(np.float32) + np.random.randn(len(z)) * 0.5

        dtype_full = [
            (attribute, "f4")
            for attribute in [
                "x",
                "y",
                "z",
                "Z_H",
                "SW",
                "AzShr",
                "Div",
                "Z_DR",
                "K_DP",
                "scale_0",
                "scale_1",
                "scale_2",
                "rot_0",
                "rot_1",
                "rot_2",
                "rot_3",
            ]
        ]
        position = np.stack((pos_x - H * 0.5 + 0.5, pos_y - W * 0.5 + 0.5, pos_z), axis=1)
        features = data[:, z, x, y].transpose(1, 0).cpu().numpy()
        scale_rot = np.zeros((len(x), 7))
        scale_rot[:, 3] = 1.0

        uncover_gaussian = np.concatenate((position, features, scale_rot), axis=1)

        # re-sample the region that is covered during the pre-reconstructionß
        indices = random.sample(range(N), N - len(x))
        gaussians_tensor = gaussians_tensor[indices]
        cover_gaussian = gaussians_tensor.numpy()

        elements = list(map(tuple, np.concatenate([cover_gaussian, uncover_gaussian], axis=0)))
        el = PlyElement.describe(np.array(elements, dtype=dtype_full), "gaussians")
        ply_data = PlyData([el])

        xoy_cam_infos, supplementary_cam_infos = self.generateRadarCameras(data.cpu().numpy())

        scene_info = SceneInfo(point_cloud=ply_data, xoy=xoy_cam_infos, supplement=supplementary_cam_infos)
        return scene_info, data[0] * 255, indices

    def generateRadarCameras(self, np_data):
        C, D, H, W = np_data.shape
        xoy_cam_infos = []
        supplementary_cam_infos = []

        # get xoy
        xoy_cam_infos = []
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        T = np.array([0, 0, 0])
        for view_depth in range(0, D):
            cam_info = CameraInfo(
                R=R, T=T, view_depth=view_depth, gt_image=get_xoy(np_data, view_depth), channel=C, width=W, height=H
            )
            xoy_cam_infos.append(cam_info)

        # get tilted xoy orthogonal to yoz
        alphas = [np.pi / 6, -np.pi / 6]
        view_depths = list(np.where(np.sum(np_data[0], axis=(0, 1)) > 100)[0])
        for alpha in alphas:
            sin_alpha = np.sin(alpha)
            cos_alpha = np.cos(alpha)
            for view_depth in random.sample(view_depths, min(self.num_vertical_samples, len(view_depths))):
                view_depth = int(view_depth - D / (2 * abs(np.tan(alpha))))
                R = np.array([[1, 0, 0], [0, cos_alpha, -sin_alpha], [0, sin_alpha, cos_alpha]])
                if alpha >= 0:
                    T = np.array(
                        [
                            0,
                            (W * 0.5 - view_depth) * cos_alpha - D / (2 * sin_alpha),
                            (view_depth - W * 0.5) * sin_alpha,
                        ]
                    )
                else:
                    T = np.array(
                        [
                            0,
                            (W * 0.5 - view_depth + D / np.tan(alpha)) * cos_alpha - D / (2 * sin_alpha),
                            (view_depth - W * 0.5 - D / np.tan(alpha)) * sin_alpha,
                        ]
                    )

                gt_image = get_tilted_xoy_orthogonal_yoz(np_data, alpha, view_depth)
                cam_info = CameraInfo(
                    R=R,
                    T=T,
                    view_depth=0,
                    gt_image=gt_image,
                    channel=C,
                    width=gt_image.shape[2],
                    height=gt_image.shape[1],
                )
                supplementary_cam_infos.append(cam_info)

        # get tilted xoy orthogonal to xoz
        alpha = [np.pi / 6, -np.pi / 6]
        view_depths = list(np.where(np.sum(np_data[0], axis=(0, 2)) > 100)[0])
        for alpha in alphas:
            sin_alpha = np.sin(alpha)
            cos_alpha = np.cos(alpha)
            for view_depth in random.sample(view_depths, min(self.num_vertical_samples, len(view_depths))):
                view_depth = int(view_depth - D / (2 * abs(np.tan(alpha))))

                R = np.array([[0, cos_alpha, -sin_alpha], [1, 0, 0], [0, sin_alpha, cos_alpha]])
                if alpha >= 0:
                    T = np.array(
                        [
                            0,
                            (H * 0.5 - view_depth) * cos_alpha - D / (2 * sin_alpha),
                            (view_depth - H * 0.5) * sin_alpha,
                        ]
                    )
                else:
                    T = np.array(
                        [
                            0,
                            (H * 0.5 - view_depth + D / np.tan(alpha)) * cos_alpha - D / (2 * sin_alpha),
                            (view_depth - H * 0.5 - D / np.tan(alpha)) * sin_alpha,
                        ]
                    )

                gt_image = get_tilted_xoy_orthogonal_xoz(np_data, alpha, view_depth)
                cam_info = CameraInfo(
                    R=R,
                    T=T,
                    view_depth=0,
                    gt_image=gt_image,
                    channel=C,
                    width=gt_image.shape[2],
                    height=gt_image.shape[1],
                )
                supplementary_cam_infos.append(cam_info)

        return xoy_cam_infos, supplementary_cam_infos

    def get_vertical_interpolation_matrix(self):
        idx2hight = {
            0: 1.5,
            1: 2,
            2: 2.5,
            3: 3,
            4: 3.5,
            5: 4,
            6: 4.5,
            7: 5,
            8: 5.5,
            9: 6,
            10: 6.5,
            11: 7,
            12: 8,
            13: 9,
            14: 10,
            15: 11,
            16: 12,
            17: 13,
            18: 14,
            19: 15,
            20: 16,
            21: 17,
            22: 18,
            23: 19,
            24: 20,
            25: 21,
            26: 22,
        }

        original_heights = np.array(list(idx2hight.values()))
        target_heights = np.arange(1.5, 22.5, 0.5)

        # find the lower and upper indices
        lower_indices = np.searchsorted(original_heights, target_heights, side="right") - 1
        upper_indices = lower_indices + 1

        # clip the indices
        lower_indices = np.clip(lower_indices, 0, len(original_heights) - 1)
        upper_indices = np.clip(upper_indices, 0, len(original_heights) - 1)

        # calculate the weights for interpolation
        lower_heights = original_heights[lower_indices]
        upper_heights = original_heights[upper_indices]
        epsilon = 1e-10
        heights_diff = upper_heights - lower_heights
        heights_diff[heights_diff == 0] = epsilon
        upper_weight = (target_heights - lower_heights) / heights_diff
        lower_weight = 1 - upper_weight

        # return the fixed interpolation matrix
        M = np.zeros((len(target_heights), len(original_heights)))
        M[np.arange(len(target_heights)), upper_indices] = upper_weight
        M[np.arange(len(target_heights)), lower_indices] = lower_weight

        return M.astype(np.float32)


if __name__ == "__main__":
    dataset = RadarDataset(max_init_points=2**14 * 3, num_vertical_samples=25)
    scene_info, data = dataset.generateRadarSceneInfo("/data/NEXRAD/nexrad_3d_v4_2_20220101T120000Z.nc")
    print(scene_info.point_cloud)
    print(scene_info.xoy)
    print(scene_info.supplement)
    print(data.shape)
    print(data)
