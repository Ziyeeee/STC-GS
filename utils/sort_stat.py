import os
from pathlib import Path
from tqdm import tqdm
import json
import h5py
import numpy as np
import torch


def float_to_int(points, scale_factor):
    return (points * scale_factor).long()


def interleave_bits(x, y, z):
    """get Morton code"""

    def spread_bits(v):
        v = (v | (v << 16) | (v << 32)) & 0x0000FF0000FF
        v = (v | (v << 8) | (v << 16)) & 0x00F00F00F00F
        v = (v | (v << 4) | (v << 8)) & 0x0C30C30C30C3
        v = (v | (v << 2) | (v << 4)) & 0x249249249249
        return v

    xx = spread_bits(x)
    yy = spread_bits(y)
    zz = spread_bits(z)

    return (xx << 2) | (yy << 1) | zz


def morton_sort(points):
    # scale float into int
    scale_factor = 50
    points[:, 0] -= points[:, 0].min()
    points[:, 1] -= points[:, 1].min()
    points[:, 2] -= points[:, 2].min()
    int_points = float_to_int(points, scale_factor)

    morton_codes = interleave_bits(int_points[:, 0], int_points[:, 1], int_points[:, 2])

    # sort by morton code
    sorted_indices = torch.argsort(morton_codes)

    return sorted_indices


def seq_str(seq_name):
    # nexrad_3d_v4_2_20220305T180000Z
    seq_str = seq_name.split("_")[-1]
    return seq_str[0:8] + seq_str[9:13]


@torch.no_grad()
def store_sorted_indices(dataset_dir: str, json_path: str):
    device = torch.device("cuda")
    with open(json_path, "r") as f:
        data = json.load(f)
    seqs = data["train"] + data["val"] + data["test"]
    print(f"Total number of gaussian squences: {len(seqs)}")

    comp_kwargs = {"compression": "gzip", "compression_opts": 4}
    hdf_path = Path(dataset_dir) / "sorted_indices.hdf5"
    print(f"Sorting and storing indices to {hdf_path}")
    with h5py.File(hdf_path, "w") as indices_f:
        for seq in seqs:
            hdf_file = os.path.join(dataset_dir, f"sequence_{seq_str(seq[0])}-{seq_str(seq[-1])}.hdf5")
            try:
                with h5py.File(hdf_file, "r") as seq_f:
                    points = seq_f[f"seq_00"][:, :3]
                    points = torch.from_numpy(points).to(device)
                    N, _ = points.shape
                    indices = morton_sort(points)
                    indices_f.create_dataset(
                        f"{seq_str(seq[0])}-{seq_str(seq[-1])}", data=indices.cpu().numpy(), **comp_kwargs
                    )
            except Exception as e:
                print(f"Cannot open {hdf_file}: {e}")
                continue
    return


def statistics(dataset_dir: str, json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)
    seqs = data["train"]

    points_mean = 0
    points_std = 0
    points_cnt = 0
    delta_mean = 0
    delta_std = 0
    delta_cnt = 0
    dataset_dir = Path(dataset_dir)
    for seq in tqdm(seqs, desc="Statistics mean:"):
        hdf_file = os.path.join(dataset_dir / f"sequence_{seq_str(seq[0])}-{seq_str(seq[-1])}.hdf5")
        try:
            with h5py.File(hdf_file, "r") as seq_f:
                seq_len = len(seq_f.keys())
                old_points = seq_f[f"seq_00"][:]
                points_mean += old_points.mean(axis=0)
                points_cnt += 1
                for i in range(1, seq_len):
                    new_points = seq_f[f"seq_{i:02d}"][:]
                    delta = new_points - old_points
                    delta_mean += delta.mean(axis=0)
                    delta_cnt += 1
                    old_points = new_points
                    points_mean += old_points.mean(axis=0)
                    points_cnt += 1

        except Exception as e:
            # print(f"Cannot open {hdf_file}: {e}")
            continue
    points_mean /= points_cnt
    delta_mean /= delta_cnt
    print(f"Points mean: {points_mean}")
    print(f"Delta mean: {delta_mean}")

    for seq in tqdm(seqs, desc="Statistics std:"):
        hdf_file = os.path.join(dataset_dir, f"sequence_{seq_str(seq[0])}-{seq_str(seq[-1])}.hdf5")
        try:
            with h5py.File(hdf_file, "r") as seq_f:
                seq_len = len(seq_f.keys())
                old_points = seq_f[f"seq_00"][:]
                points_std += ((old_points - points_mean) ** 2).mean(axis=0)
                for i in range(1, seq_len):
                    new_points = seq_f[f"seq_{i:02d}"][:]
                    delta = new_points - old_points
                    delta_std += ((delta - delta_mean) ** 2).mean(axis=0)
                    old_points = new_points
                    points_std += ((old_points - points_mean) ** 2).mean(axis=0)

        except Exception as e:
            # print(f"Cannot open {hdf_file}: {e}")
            continue
    points_std = np.sqrt(points_std / points_cnt)
    delta_std = np.sqrt(delta_std / delta_cnt)
    print(f"Points std: {points_std}")
    print(f"Delta std: {delta_std}")

    # save statistics
    points_mean = np.zeros_like(points_mean)
    points_mean[2] = 18.0
    points_std[0:3] = [64.0, 64.0, 4.0]
    delta_mean = np.zeros_like(delta_mean)

    np.savez(
        dataset_dir / "statistics.npz", mean=points_mean, std=points_std, delta_mean=delta_mean, delta_std=delta_std
    )
    return


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--json_path", type=str, required=True)
    args = parser.parse_args()

    store_sorted_indices(args.dataset_dir, args.json_path)
    statistics(args.dataset_dir, args.json_path)
