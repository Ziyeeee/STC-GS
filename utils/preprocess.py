from pathlib import Path
from datetime import datetime
import h5py
import json
import random


def seq_str(seq_name):
    # nexrad_3d_v4_2_20220305T180000Z
    seq_str = seq_name.split("_")[-1]
    return seq_str[0:8] + seq_str[9:13]


def continuity_check(h5_path):
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        keys.sort()

    # find continuous sequence
    # nexrad_3d_v4_2_20220305T180000Z, nexrad_3d_v4_2_20220305T180500Z, nexrad_3d_v4_2_20220305T181000Z
    seqs = []
    temp_seq = [keys[0]]
    old_frame = datetime.strptime(seq_str(keys[0]), "%Y%m%d%H%M")
    for key in keys[1:]:
        new_frame = datetime.strptime(seq_str(key), "%Y%m%d%H%M")
        if (new_frame - old_frame).total_seconds() == 300:
            temp_seq.append(key)
        else:
            if len(temp_seq) >= 25:
                seqs.append(temp_seq)
            temp_seq = [key]
        old_frame = new_frame

    if len(temp_seq) >= 25:
        seqs.append(temp_seq)

    print(f"Found {len(seqs)} events.")
    return seqs


def partition_sequence(seqs, seq_len=25, stride=10):
    frame_cnt = 0
    for seq in seqs:
        frame_cnt += len(seq)
    print(f"Total {frame_cnt} frames.")

    partitioned_sequence = []
    for seq in seqs:
        length = len(seq)
        for i in range(0, length - seq_len, stride):
            partitioned_sequence.append(seq[i : i + seq_len])
    print(f"Partitioned {len(partitioned_sequence)} sequences.")
    return partitioned_sequence


def split_dataset(seqs, path):
    train_len = int(len(seqs) * 0.9)
    val_len = int(len(seqs) * 0.05)

    random.shuffle(seqs)
    trian_data = seqs[:train_len]
    val_data = seqs[train_len : train_len + val_len]
    test_data = seqs[train_len + val_len :]
    trian_data.sort()
    val_data.sort()
    test_data.sort()

    print(f"Train: {len(trian_data)} Val: {len(val_data)} Test: {len(test_data)}")

    dataset = {"train": trian_data, "val": val_data, "test": test_data}
    with open(str(path).replace("hdf5", "json"), "w") as f:
        json.dump(dataset, f, indent=4)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    path = Path(args.path)
    seqs = continuity_check(path)
    partitioned_sequence = partition_sequence(seqs)
    split_dataset(partitioned_sequence, path)
