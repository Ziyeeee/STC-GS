import os
import sys
import time
import json
import h5py
from pathlib import Path
import logging
import logging.handlers
import torch.multiprocessing as multiprocessing
import torch
import torch.nn.functional as F
import numpy as np
from random import randint


from argparse import ArgumentParser, Namespace
from arguments import ModelParams, OptimizationParams

from radar_gs.render import render
from radar_gs.gaussian_model import GaussianModel, DeltaGaussianModel
from radar_gs.camera import cameraList_from_camInfos
from radar_gs.radar_dataset import RadarDataset
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, l2_loss, exp_l2_loss, SSIM, Energy

from flow.core.raft import RAFT


try:
    from torch.utils.tensorboard.writer import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def prepare_output_and_logger(log_dir):
    # Set up output folder

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(log_dir)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def set_output_dir(dataset_path):
    gaussians_dir = dataset_path.parent / "gaussians"
    logs_dir = dataset_path.parent / "logs"
    return gaussians_dir, logs_dir


def listener(log_queue, log_file):
    # Set up logging
    root = logging.getLogger()
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Process log messages
    while True:
        try:
            if not log_queue.empty():
                record = log_queue.get()
                if record is None:
                    break
                logger = logging.getLogger(record.name)
                logger.handle(record)
        except Exception:
            import sys
            import traceback

            print("Logging error:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def reconstruct(args):
    hdf_path = Path(args.hdf_path)
    gaussians_dir, logs_dir = set_output_dir(hdf_path)
    gaussians_dir.mkdir(parents=False, exist_ok=True)
    logs_dir.mkdir(parents=False, exist_ok=True)

    seq_len = args.seq_len
    with open(args.json_path, "r") as f:
        data = json.load(f)
    seqs = data["train"] + data["val"] + data["test"]

    with open(logs_dir / "cfg_args", "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    manager = multiprocessing.Manager()
    pid2device = manager.dict()
    log_queue = manager.Queue()
    log_file = logs_dir / "__log__.txt"

    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None:
        num_gpus = len(visible_devices.split(","))
    else:
        num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    listener_process = multiprocessing.Process(target=listener, args=(log_queue, log_file))
    listener_process.start()

    if args.debug:
        # modify args for faster debugging
        args.num_processes = 1
        args.frame_iterations = 1500
        args.energy_iterations = 500
        args.seq_len = 10
        opt_params = op.extract(args)

        dataset = RadarDataset(
            max_init_points=args.max_init_points,
            num_vertical_samples=args.num_vertical_samples,
            seq_len=seq_len,
        )
        mission_idx = 0
        for seq_info in seqs:
            reconstruct_one(
                args,
                opt_params,
                seq_info[: args.seq_len],
                dataset,
                mission_idx,
                len(seqs),
                pid2device,
                log_queue,
            )
            mission_idx += 1

    else:
        opt_params = op.extract(args)
        with multiprocessing.Pool(processes=args.num_processes) as pool:
            mission_idx = 0
            dataset = RadarDataset(
                max_init_points=args.max_init_points,
                num_vertical_samples=args.num_vertical_samples,
                seq_len=seq_len,
            )
            for seq_info in seqs:
                pool.apply_async(
                    reconstruct_one,
                    (
                        args,
                        opt_params,
                        seq_info,
                        dataset,
                        mission_idx,
                        len(seqs),
                        pid2device,
                        log_queue,
                    ),
                    error_callback=lambda e: print(e),
                )
                mission_idx += 1
            pool.close()
            pool.join()

    log_queue.put(None)
    listener_process.join()


def seq_str(seq_name):
    # nexrad_3d_v4_2_20220305T180000Z
    seq_str = seq_name.split("_")[-1]
    return seq_str[0:8] + seq_str[9:13]


@torch.no_grad()
def get_flow(xyz, raft, frame_old, frame_new, ema_flow_xy, ema_flow_zy, ema_flow_zx):
    D, H, W = frame_old.shape
    frame_old_xy = frame_old[: (D // 3) * 3].reshape(D // 3, 3, H, W).mean(dim=0, keepdim=True)
    frame_new_xy = frame_new[: (D // 3) * 3].reshape(D // 3, 3, H, W).mean(dim=0, keepdim=True)
    _, flow_xy = raft(frame_old_xy, frame_new_xy, iters=30, test_mode=True)
    flow_xy = torch.stack([flow_xy[0, 1], flow_xy[0, 0]], dim=-1)
    D, H, W = frame_old.shape
    frame_old_zy = frame_old.reshape(D, H // 32, 32, W).permute(1, 2, 0, 3).mean(dim=1, keepdim=True)
    frame_new_zy = frame_new.reshape(D, H // 32, 32, W).permute(1, 2, 0, 3).mean(dim=1, keepdim=True)
    frame_old_zy = torch.cat(
        [
            torch.zeros(
                (H // 32, 1, (128 - D) // 2, W),
                dtype=frame_old_xy.dtype,
                device=frame_old_xy.device,
            ),
            frame_old_zy,
            torch.zeros(
                (H // 32, 1, (128 - D) // 2, W),
                dtype=frame_old_xy.dtype,
                device=frame_old_xy.device,
            ),
        ],
        dim=2,
    )
    frame_new_zy = torch.cat(
        [
            torch.zeros(
                (H // 32, 1, (128 - D) // 2, W),
                dtype=frame_new_xy.dtype,
                device=frame_new_xy.device,
            ),
            frame_new_zy,
            torch.zeros(
                (H // 32, 1, (128 - D) // 2, W),
                dtype=frame_new_xy.dtype,
                device=frame_new_xy.device,
            ),
        ],
        dim=2,
    )
    _, flow_zy = raft(
        frame_old_zy.repeat(1, 3, 1, 1),
        frame_new_zy.repeat(1, 3, 1, 1),
        iters=30,
        test_mode=True,
    )
    flow_zy = flow_zy[:, 1, (128 - D) // 2 : (128 - D) // 2 + D]

    frame_old_zx = frame_old.reshape(D, H, W // 32, 32).permute(2, 3, 0, 1).mean(dim=1, keepdim=True)
    frame_new_zx = frame_new.reshape(D, H, W // 32, 32).permute(2, 3, 0, 1).mean(dim=1, keepdim=True)
    frame_old_zx = torch.cat(
        [
            torch.zeros(
                (W // 32, 1, (128 - D) // 2, H),
                dtype=frame_old_xy.dtype,
                device=frame_old_xy.device,
            ),
            frame_old_zx,
            torch.zeros(
                (W // 32, 1, (128 - D) // 2, H),
                dtype=frame_old_xy.dtype,
                device=frame_old_xy.device,
            ),
        ],
        dim=2,
    )
    frame_new_zx = torch.cat(
        [
            torch.zeros(
                (W // 32, 1, (128 - D) // 2, H),
                dtype=frame_new_xy.dtype,
                device=frame_new_xy.device,
            ),
            frame_new_zx,
            torch.zeros(
                (W // 32, 1, (128 - D) // 2, H),
                dtype=frame_new_xy.dtype,
                device=frame_new_xy.device,
            ),
        ],
        dim=2,
    )
    _, flow_zx = raft(
        frame_old_zx.repeat(1, 3, 1, 1),
        frame_new_zx.repeat(1, 3, 1, 1),
        iters=30,
        test_mode=True,
    )
    flow_zx = flow_zx[:, 1, (128 - D) // 2 : (128 - D) // 2 + D]

    ema_flow_xy = 0.2 * flow_xy + 0.8 * ema_flow_xy if ema_flow_xy is not None else flow_xy
    ema_flow_zy = 0.2 * flow_zy + 0.8 * ema_flow_zy if ema_flow_zy is not None else flow_zy
    ema_flow_zx = 0.2 * flow_zx + 0.8 * ema_flow_zx if ema_flow_zx is not None else flow_zx

    D, H, W = frame_new.shape
    x = torch.clamp(xyz[:, 0] + H * 0.5 - 0.5, 0, H - 1)
    y = torch.clamp(xyz[:, 1] + W * 0.5 - 0.5, 0, W - 1)
    z = torch.clamp(xyz[:, 2], 0, D - 1)
    zx = torch.clamp(torch.round(x / 32), 0, H // 32 - 1).long()
    zy = torch.clamp(torch.round(y / 32), 0, W // 32 - 1).long()
    x = torch.round(x).long()
    y = torch.round(y).long()
    z = torch.round(z).long()
    flow_xy = ema_flow_xy[x, y]
    flow_z = (ema_flow_zy[zx, z, y] + ema_flow_zx[zy, z, x]) / 2
    flow_xyz = torch.cat([flow_xy, flow_z.unsqueeze(-1)], dim=-1)
    return flow_xyz


def reconstruct_one(args, opt, seq_info, dataset, mission_idx, total_seq, pid2device, log_queue):
    import torch

    pid = os.getpid()
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None:
        num_gpus = len(visible_devices.split(","))
    else:
        num_gpus = torch.cuda.device_count()

    # logger.info(visible_devices)
    if pid in pid2device:
        device_idx = pid2device[pid]
    else:
        device_idx = mission_idx % num_gpus
        pid2device[pid] = device_idx
    device = torch.device(f"cuda:{device_idx}")
    torch.cuda.set_device(torch.device(f"cuda:{device_idx}"))

    logger = logging.getLogger(f"worker_{pid}")
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        handler = logging.handlers.QueueHandler(log_queue)
        logger.addHandler(handler)

    hdf_path = Path(args.hdf_path)
    gaussians_dir, logs_dir = set_output_dir(hdf_path)

    gaussian_path = gaussians_dir / f"sequence_{seq_str(seq_info[0])}-{seq_str(seq_info[-1])}.hdf5"
    f = h5py.File(gaussian_path, "w")
    f.attrs["seq_info"] = ", ".join(seq_info)
    comp_kwargs = {"compression": "gzip", "compression_opts": 4}
    log_dir = logs_dir / f"sequence_{seq_str(seq_info[0])}-{seq_str(seq_info[-1])}"
    log_dir.mkdir(parents=False, exist_ok=True)
    tb_writer = prepare_output_and_logger(log_dir)

    start_time = time.time()
    seq_len = len(seq_info)
    scene_info, frame_old = dataset.generateRadarSceneInfo(hdf_path, seq_info[-1], device, init_ply=True)
    gaussians = GaussianModel(device)
    gaussians.init_points(scene_info.point_cloud)
    gaussians.training_setup(opt, max_iteration=opt.frame_iterations)

    # Load RAFT model for flow estimation
    raft = torch.nn.DataParallel(RAFT(args))
    raft.load_state_dict(torch.load(args.raft_model, weights_only=False))
    raft = raft.module
    raft.to(device)
    raft.eval()

    ssim = SSIM(channel=args.radar_channel).to(device)
    get_energy = Energy(window_size=11, sigm=torch.e, device=device)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    print(
        f"[{mission_idx+1}/{total_seq}] Reconstruct sequence_{seq_info[0]}-{seq_info[-1]}, on device cuda:{device_idx}"
    )
    logger.info(
        f"[{mission_idx+1}/{total_seq}] Reconstruct sequence_{seq_info[0]}-{seq_info[-1]}, on device cuda:{device_idx}"
    )

    global_iteration = 0
    # pre-reconstruction (backward)
    ema_flow_xy, ema_flow_zy, ema_flow_zx = None, None, None
    inverse_d_gaussians = []
    for seq_idx in range(seq_len - 1, -1, -1):
        print(f"Reconstruct [{-seq_idx-1}/{seq_len}] in sequence_{seq_info[0]}-{seq_info[-1]}")
        logger.info(f"Reconstruct [{-seq_idx-1}/{seq_len}] in sequence_{seq_info[0]}-{seq_info[-1]}")
        if seq_idx < seq_len - 1:
            scene_info, frame_new = dataset.generateRadarSceneInfo(hdf_path, seq_info[seq_idx], device, init_ply=False)
        scene_info.set_xoy(cameraList_from_camInfos(scene_info.xoy))
        scene_info.set_supplement(cameraList_from_camInfos(scene_info.supplement))

        xoy_viewpoint_stack = None
        if seq_len - seq_idx == 1:
            d_gaussians = None
        else:
            if seq_len - seq_idx == 2:
                d_gaussians = DeltaGaussianModel(gaussians)
                d_gaussians.training_setup(opt, max_iteration=opt.frame_iterations)
            else:
                d_gaussians.reset()
                d_gaussians.training_setup(opt, max_iteration=opt.frame_iterations)

            xyz = gaussians.get_xyz.clone().detach()
            flow_xyz = get_flow(xyz, raft, frame_old, frame_new, ema_flow_xy, ema_flow_zy, ema_flow_zx)

            frame_old = frame_new

        for iteration in range(1, opt.energy_iterations + 1):
            global_iteration += 1
            iter_start.record()

            # Pick a random Camera
            if not xoy_viewpoint_stack:
                xoy_viewpoint_stack = scene_info.xoy.copy()
            viewpoint_cam = xoy_viewpoint_stack.pop(randint(0, len(xoy_viewpoint_stack) - 1))

            # Render
            image = render(viewpoint_cam, gaussians, d_gaussians, energy=True)["render"]
            gt_image = viewpoint_cam.gt_image.to(device)
            gt_image = get_energy(gt_image[0].unsqueeze(dim=0)).squeeze(dim=0)
            gt_image = torch.clamp(gt_image, 0.0, 0.3)
            Ll2 = l2_loss(image, gt_image)
            if d_gaussians is not None:
                flow_loss = F.mse_loss(d_gaussians.get_dxyz, flow_xyz)
            else:
                flow_loss = 0.0
            loss = Ll2 + 0.02 * flow_loss

            loss.backward()
            iter_end.record()

            with torch.no_grad():
                if seq_len - seq_idx == 1:
                    g_lr = gaussians.update_learning_rate(iteration, energy=True)
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    g_lr = d_gaussians.update_learning_rate(iteration, energy=True)
                    d_gaussians.optimizer.step()
                    d_gaussians.optimizer.zero_grad(set_to_none=True)
        if tb_writer is not None:
            sequence_report(tb_writer, -seq_idx - 1, gaussians, d_gaussians, scene_info)
        if seq_len - seq_idx > 1:
            gaussians.updata_gaussians(d_gaussians)
            inverse_d_gaussians.append(d_gaussians)

    # prepare gaussians for reconstruction
    cam_infos = scene_info.xoy.copy()
    energy_list = []
    for idx, cam_info in enumerate(cam_infos):
        energy_img = render(cam_info, gaussians, None, energy=True)["render"].squeeze(dim=0)
        energy_list.append(energy_img)
    energy_img = torch.stack(energy_list, dim=0)
    gaussians_tensor = gaussians.get_gaussians_as_tensor().cpu()
    scene_info, frame_old, indices = dataset.generateRadarSceneInfoWithInverseGaussian(
        hdf_path, seq_info[0], gaussians_tensor, energy_img, device
    )
    gaussians = GaussianModel(device)
    gaussians.init_points(scene_info.point_cloud)
    gaussians.training_setup(opt, max_iteration=opt.frame_iterations)

    # reconstruction (forward)
    ema_flow_xy, ema_flow_zy, ema_flow_zx = None, None, None
    for seq_idx in range(0, seq_len):
        print(f"Reconstruct [{seq_idx+1}/{seq_len}] in sequence_{seq_info[0]}-{seq_info[-1]}")
        logger.info(f"Reconstruct [{seq_idx+1}/{seq_len}] in sequence_{seq_info[0]}-{seq_info[-1]}")
        if seq_idx > 0:
            scene_info, frame_new = dataset.generateRadarSceneInfo(hdf_path, seq_info[seq_idx], device, init_ply=False)
        scene_info.set_xoy(cameraList_from_camInfos(scene_info.xoy))
        scene_info.set_supplement(cameraList_from_camInfos(scene_info.supplement))

        xoy_viewpoint_stack = None
        supplement_viewpoint_stack = None
        if seq_idx == 0:
            d_gaussians = None
        else:
            if seq_idx == 1:
                d_gaussians = DeltaGaussianModel(
                    gaussians,
                    inverse_d_gaussians[0],
                    indices,
                    max_points=dataset.max_init_points,
                )
                d_gaussians.training_setup(opt, max_iteration=opt.frame_iterations)
            else:
                d_gaussians.reset(
                    inverse_d_gaussians[seq_idx - 1],
                    indices,
                    max_points=dataset.max_init_points,
                )
                d_gaussians.training_setup(opt, max_iteration=opt.frame_iterations)

            xyz = gaussians.get_xyz.clone().detach()
            flow_xyz = get_flow(xyz, raft, frame_old, frame_new, ema_flow_xy, ema_flow_zy, ema_flow_zx)
            frame_old = frame_new

        for iteration in range(1, opt.frame_iterations + 1):
            global_iteration += 1
            iter_start.record()

            # Pick a random Camera
            if not xoy_viewpoint_stack:
                xoy_viewpoint_stack = scene_info.xoy.copy()
            if not supplement_viewpoint_stack:
                supplement_viewpoint_stack = scene_info.supplement.copy()

            if iteration % 2 == 0:
                viewpoint_cam = supplement_viewpoint_stack.pop(randint(0, len(supplement_viewpoint_stack) - 1))
            else:
                viewpoint_cam = xoy_viewpoint_stack.pop(randint(0, len(xoy_viewpoint_stack) - 1))

            # Render
            if iteration <= opt.energy_iterations:
                image = render(viewpoint_cam, gaussians, d_gaussians, energy=True)["render"]
                gt_image = viewpoint_cam.gt_image.to(device)
                gt_image = get_energy(gt_image[0].unsqueeze(dim=0))
                gt_image = torch.clamp(gt_image, 0.0, 0.3)
                Ll2 = l2_loss(image, gt_image)
                if d_gaussians is not None:
                    flow_loss = F.mse_loss(d_gaussians.get_dxyz, flow_xyz)
                else:
                    flow_loss = 0.0
                loss = Ll2 + 0.01 * flow_loss
            else:
                image = render(viewpoint_cam, gaussians, d_gaussians, energy=False)["render"]
                gt_image = viewpoint_cam.gt_image.to(device)
                Ll2 = exp_l2_loss(image, gt_image)
                ssim_loss = 1.0 - ssim(image, gt_image)
                if d_gaussians is not None:
                    flow_loss = F.mse_loss(d_gaussians.get_dxyz, flow_xyz)
                else:
                    flow_loss = 0.0
                loss = (1.0 - opt.lambda_dssim) * Ll2 + opt.lambda_dssim * ssim_loss + 0.01 * flow_loss

            loss.backward()
            iter_end.record()

            with torch.no_grad():
                if seq_idx == 0:
                    g_lr = gaussians.update_learning_rate(
                        iteration, energy=True if iteration < opt.energy_iterations else False
                    )
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    g_lr = d_gaussians.update_learning_rate(
                        iteration, energy=True if iteration < opt.energy_iterations else False
                    )
                    d_gaussians.optimizer.step()
                    d_gaussians.optimizer.zero_grad(set_to_none=True)
        if tb_writer is not None:
            sequence_report(tb_writer, seq_idx + 1, gaussians, d_gaussians, scene_info)
        if seq_idx > 0:
            gaussians.updata_gaussians(d_gaussians)
        if seq_idx >= 0:
            data = gaussians.get_gaussians_as_tensor()
            f.create_dataset(f"seq_{seq_idx:0>2d}", data=data.detach().cpu().numpy(), **comp_kwargs)

    torch.cuda.empty_cache()
    print(f"[{mission_idx+1}/{total_seq}] Done. Takes {(time.time()-start_time)/60:.2f}min")
    logger.info(f"[{mission_idx+1}/{total_seq}] Done. Takes {(time.time()-start_time)/60:.2f}min")
    if tb_writer is not None:
        tb_writer.close()
    f.close()
    return


def sequence_report(tb_writer, seq_idx, gaussians, d_gaussians, scene_info):
    cam_infos = scene_info.xoy.copy()
    l1_test = 0.0
    for idx, cam_info in enumerate(cam_infos):
        image = render(cam_info, gaussians, d_gaussians)["render"]
        gt_image = cam_info.gt_image.to(image.device)
        l1_test += l1_loss(image, gt_image)
        # if idx in [0, 4, 8]:
        if (seq_idx in [-25, -20, -15, -10, -5, -1, 1, 5, 10, 15, 20, 25]) and (idx in [0, 4, 8]):
            energy_img = render(cam_info, gaussians, d_gaussians, energy=True)["render"].squeeze(dim=0)
            tb_writer.add_images(
                "seq/view_{}_energy".format(idx),
                energy_img.detach().cpu().numpy(),
                global_step=seq_idx,
                dataformats="HW",
            )
            image = image[0].detach().cpu().numpy()
            gt_image = gt_image[0].detach().cpu().numpy()
            if seq_idx >= 0:
                tb_writer.add_images(
                    "seq/view_{}_render".format(idx),
                    image,
                    global_step=seq_idx,
                    dataformats="HW",
                )
            tb_writer.add_images(
                "seq/view_{}_gt".format(idx),
                gt_image,
                global_step=seq_idx,
                dataformats="HW",
            )

    l1_test /= len(cam_infos)
    tb_writer.add_scalar("sequence_l1", l1_test, seq_idx)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--hdf_path", type=str, required=True)
    # parser.add_argument("--json_path", type=str)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--raft_model", help="restore checkpoint", default="flow/models/raft-things.pth")
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument(
        "--alternate_corr",
        action="store_true",
        help="use efficent correlation implementation",
    )
    args = parser.parse_args(sys.argv[1:])
    args.json_path = args.hdf_path.replace("hdf5", "json")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    reconstruct(args)

    # All done
    print("\nReconstruction complete.")
