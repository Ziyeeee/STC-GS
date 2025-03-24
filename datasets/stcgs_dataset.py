import os
import sys
import time
from tqdm import tqdm
from pathlib import Path
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn


def seq_str(seq_name):
    # nexrad_3d_v4_2_20220305T180000Z
    seq_str = seq_name.split("_")[-1]
    return seq_str[0:8] + seq_str[9:13]


class GaussiansDataset(Dataset):
    def __init__(self, dataset_dir: str, json_path: str, mode: str, seq_len=25, hdf_path="") -> None:
        super().__init__()
        assert mode in ["train", "val", "test"]

        with open(json_path, "r") as f:
            data = json.load(f)
            seqs = data[mode]
            self.seqs = seqs

        dataset_dir = Path(dataset_dir)
        self.indices_path = dataset_dir / "sorted_indices.hdf5"
        statistic = np.load(dataset_dir / "statistics.npz")
        self.statistic = {
            "mean": torch.from_numpy(statistic["mean"]).float(),
            "std": torch.from_numpy(statistic["std"]).float(),
            "delta_mean": torch.from_numpy(statistic["delta_mean"]).float(),
            "delta_std": torch.from_numpy(statistic["delta_std"]).float(),
        }
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.seq_len = seq_len
        self.hdf_path = hdf_path
        self.M = get_vertical_interpolation_matrix()

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, index: int):
        seq = self.seqs[index]

        with h5py.File(self.indices_path, "r") as indice_f:
            indices = indice_f[f"{seq_str(seq[0])}-{seq_str(seq[-1])}"]

            hdf_file = os.path.join(self.dataset_dir, f"sequence_{seq_str(seq[0])}-{seq_str(seq[-1])}.hdf5")
            with h5py.File(hdf_file, "r") as data_f:
                seq_data = []
                for idx in range(self.seq_len):
                    try:
                        data = data_f[f"seq_{idx:0>2d}"][:]
                        data = data[indices]
                        seq_data.append(data)
                    except Exception as e:
                        print(f"Read seq_idx {idx} in {seq_str(seq[0])}-{seq_str(seq[-1])} failed, with error {str(e)}")
                        break

        if len(seq_data) > 1:
            seq_data = np.stack(seq_data, axis=0)
        else:
            seq_data = seq_data[0]

        seq_data = torch.from_numpy(seq_data).float()

        if self.mode == "test":
            data_list = []
            keys = ["Z_H", "SW", "AzShr", "Div", "Z_DR", "K_DP"]
            with h5py.File(self.hdf_path, "r") as f:
                for item_name in seq:
                    group = f[item_name]

                    data = np.stack([group[key][:] for key in keys], axis=0)
                    data = data[:, 2:]
                    data = data / np.array([70, 20, 0.02, 0.02, 20, 50]).reshape(-1, 1, 1, 1)
                    data = np.tensordot(self.M, data, axes=([1], [1]))
                    data = np.clip(data, -1, 1).transpose(1, 0, 2, 3).astype(np.float32)
                    data_list.append(data)

            data = np.stack(data_list, axis=0)
            data = torch.from_numpy(data).float()
            return seq_data, data, seq
        else:
            return seq_data

    def norm_points(self, data):
        return (data - self.statistic["mean"].to(data.device)) / self.statistic["std"].to(data.device)

    def denorm_points(self, data):
        return data * self.statistic["std"].to(data.device) + self.statistic["mean"].to(data.device)

    def norm_delta(self, data):
        return (data - self.statistic["delta_mean"].to(data.device)) / self.statistic["delta_std"].to(data.device)

    def denorm_delta(self, data):
        return data * self.statistic["delta_std"].to(data.device) + self.statistic["delta_mean"].to(data.device)


def get_vertical_interpolation_matrix():
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
    target_heights = np.arange(1.5, 19.5, 0.5)

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
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="")
    parser.add_argument("--json_path", type=str, default="")
    parser.add_argument("--hdf_path", type=str, default="")
    args = parser.parse_args()

    dataset = GaussiansDataset(args.dataset_dir, args.json_path, mode="train")
    print(len(dataset))
    data = dataset[0]
    L, N, C = data.shape
    norm_data = dataset.norm_points(data)
    denorm_data = dataset.denorm_points(norm_data)
    print(f"Data shape: {data.shape}")
    print(f"Check norm and denorm: {(denorm_data - data).abs().max() < 1e-5}")
    for data in tqdm(dataset):
        pass
    dataset = GaussiansDataset(args.dataset_dir, args.json_path, mode="val")
    for data in tqdm(dataset):
        pass
    dataset = GaussiansDataset(args.dataset_dir, args.json_path, mode="test", hdf_path=args.hdf_path)
    for data in tqdm(dataset):
        pass
