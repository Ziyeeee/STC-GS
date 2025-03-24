import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
from tqdm import tqdm
import lpips

from utils.lpips_radar import MyLPIPS as lpips_radar


class ME(nn.Module):
    def __init__(self):
        super(ME, self).__init__()
        self.cnt = 0
        self.loss = 0.0

    def forward(self, pre, label):
        self.cnt += 1
        if label.ndim == 5:
            self.loss += torch.mean(pre - label, dim=[0, 2, 3, 4])
        elif label.ndim == 6:
            self.loss += torch.mean(pre - label, dim=[0, 2, 3, 4, 5])

    def calculate(self):
        return self.loss / self.cnt


class MAE(nn.Module):
    def __init__(self):
        super(MAE, self).__init__()
        self.cnt = 0
        self.loss = 0.0

    def forward(self, pre, label):
        self.cnt += 1
        if label.ndim == 5:
            self.loss += torch.mean(torch.abs(pre - label), dim=[0, 2, 3, 4])
        elif label.ndim == 6:
            self.loss += torch.mean(torch.abs(pre - label), dim=[0, 2, 3, 4, 5])

    def calculate(self):
        return self.loss / self.cnt


class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()
        self.cnt = 0
        self.loss = 0.0

    def forward(self, pre, label):
        self.cnt += 1
        if label.ndim == 5:
            self.loss += torch.mean(torch.square(pre - label), dim=[0, 2, 3, 4])
        elif label.ndim == 6:
            self.loss += torch.mean(torch.square(pre - label), dim=[0, 2, 3, 4, 5])

    def calculate(self):
        return torch.sqrt(self.loss / self.cnt)


class CC(nn.Module):
    def __init__(self):
        super(CC, self).__init__()
        self.cnt = 0
        self.loss = 0.0

    def forward(self, pre, label):
        B, L = pre.shape[0], pre.shape[1]
        self.cnt += B
        for b in range(B):
            loss = []
            for l in range(L):
                temp_pred = pre[b, l]
                temp_label = label[b, l]
                mask = (torch.abs(temp_pred) > 1e-3) | (torch.abs(temp_label) > 1e-3)
                temp_pred = temp_pred[mask]
                temp_label = temp_label[mask]
                pred_mean = torch.mean(temp_pred)
                label_mean = torch.mean(temp_label)
                cc = torch.sum((temp_pred - pred_mean) * (temp_label - label_mean)) / torch.sqrt(
                    torch.sum(torch.square(temp_pred - pred_mean)) * torch.sum(torch.square(temp_label - label_mean))
                )
                loss.append(cc)
            self.loss += torch.stack(loss)

    def calculate(self):
        return self.loss / self.cnt


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(2)
    _3D_window = torch.matmul(_2D_window, _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window


class SSIM3D(torch.nn.Module):
    def __init__(self, window_size=11):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, img1, img2):
        if img1.is_cuda:
            self.window = self.window.cuda(img1.get_device())
        self.window = self.window.type_as(img1)

        inputs = [img1, img2, img1 * img1, img2 * img2, img1 * img2]
        inputs = torch.stack(inputs, dim=0)
        outputs = F.conv3d(inputs, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1 = outputs[0]
        mu2 = outputs[1]

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = outputs[2] - mu1_sq
        sigma2_sq = outputs[3] - mu2_sq
        sigma12 = outputs[4] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / (
            (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        )

        return ssim_map.mean()


class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.cnt = 0
        self.loss = 0.0
        self.SSIM = SSIM3D()

    def forward(self, pre, label):
        B, L, D, H, W = pre.shape
        self.cnt += B
        for b in range(B):
            loss = []
            for l in range(L):
                loss.append(self.SSIM(pre[b, l, None], label[b, l, None]))
            self.loss += torch.stack(loss)

    def calculate(self):
        return self.loss / self.cnt


class LPIPS(nn.Module):
    def __init__(self):
        super(LPIPS, self).__init__()
        self.cnt = 0
        self.loss = 0.0
        self.lpips = lpips.LPIPS(net="alex")

    def forward(self, pred, label, norm=False):
        self.lpips = self.lpips.to(pred.device)
        B, L, D, H, W = pred.shape
        self.cnt += B * D
        if not norm:
            pred = pred * 2 - 1
            label = label * 2 - 1
        pred = pred.permute(0, 2, 1, 3, 4).reshape(B * D, L, H, W)
        label = label.permute(0, 2, 1, 3, 4).reshape(B * D, L, H, W)
        loss = []
        for l in range(L):
            loss.append(
                self.lpips.forward(pred[:, l, None].repeat(1, 3, 1, 1), label[:, l, None].repeat(1, 3, 1, 1)).sum()
            )
        self.loss += torch.stack(loss)

    def calculate(self):
        return self.loss / self.cnt


class MyLPIPS(nn.Module):
    def __init__(self):
        super(MyLPIPS, self).__init__()
        self.cnt = 0
        self.loss = 0.0
        self.lpips = lpips_radar("alexnet.pth")

    def forward(self, pred, label):
        self.lpips = self.lpips.to(pred.device)
        B, L, D, H, W = pred.shape
        self.cnt += B * D
        pred = pred.permute(0, 2, 1, 3, 4).reshape(B * D, L, H, W)
        label = label.permute(0, 2, 1, 3, 4).reshape(B * D, L, H, W)
        loss = []
        for l in range(L):
            loss.append(self.lpips.forward(pred[:, l, None], label[:, l, None]).sum())
        self.loss += torch.stack(loss)

    def calculate(self):
        return self.loss / self.cnt


class BinaryEvaluator(nn.Module):
    def __init__(self, seq_len, thresholds=[20, 30, 40]):
        super(BinaryEvaluator, self).__init__()
        self.metrics = {}
        self.thresholds = thresholds
        for threshold in self.thresholds:
            self.metrics[threshold] = {
                "TP": [0.0 for _ in range(seq_len)],
                "FN": [0.0 for _ in range(seq_len)],
                "FP": [0.0 for _ in range(seq_len)],
                "TP_4x4": [0.0 for _ in range(seq_len)],
                "FN_4x4": [0.0 for _ in range(seq_len)],
                "FP_4x4": [0.0 for _ in range(seq_len)],
                "TP_16x16": [0.0 for _ in range(seq_len)],
                "FN_16x16": [0.0 for _ in range(seq_len)],
                "FP_16x16": [0.0 for _ in range(seq_len)],
            }

        self.seq_len = seq_len

    @torch.no_grad()
    def forward(self, pred, gt):
        print(f"Ground Truth max: {gt.max()}, min: {gt.min()}")
        print(f"Prediction max: {pred.max()}, min: {pred.min()}")

        assert pred.shape == gt.shape, f"pred_batch.shape: {pred.shape}, true_batch.shape: {gt.shape}"

        for t in range(self.seq_len):
            for threshold in self.thresholds:
                scores = self.cal_frame(gt[:, t], pred[:, t], threshold)
                self.metrics[threshold]["TP"][t] += scores["TP"]
                self.metrics[threshold]["FN"][t] += scores["FN"]
                self.metrics[threshold]["FP"][t] += scores["FP"]

                score = self.cal_frame(F.max_pool2d(gt[:, t], 4), F.max_pool2d(pred[:, t], 4), threshold)
                self.metrics[threshold]["TP_4x4"][t] += score["TP"]
                self.metrics[threshold]["FN_4x4"][t] += score["FN"]
                self.metrics[threshold]["FP_4x4"][t] += score["FP"]

                score = self.cal_frame(F.max_pool2d(gt[:, t], 16), F.max_pool2d(pred[:, t], 16), threshold)
                self.metrics[threshold]["TP_16x16"][t] += score["TP"]
                self.metrics[threshold]["FN_16x16"][t] += score["FN"]
                self.metrics[threshold]["FP_16x16"][t] += score["FP"]

    def cal_frame(self, obs, sim, threshold):
        obs = torch.where(obs >= threshold, 1, 0)
        sim = torch.where(sim >= threshold, 1, 0)

        # True positive (TP)
        TP = (obs == 1) & (sim == 1)
        TP = torch.sum(TP).item()

        # False negative (FN)
        FN = (obs == 1) & (sim == 0)
        FN = torch.sum(FN).item()

        # False positive (FP)
        FP = (obs == 0) & (sim == 1)
        FP = torch.sum(FP).item()

        return {
            "TP": TP,
            "FN": FN,
            "FP": FP,
        }

    def calculate(self):
        score_dict = {}
        for threshold in self.thresholds:
            score_dict[threshold] = {}
        for threshold in self.thresholds:
            TP = torch.tensor(self.metrics[threshold]["TP"])
            FN = torch.tensor(self.metrics[threshold]["FN"])
            FP = torch.tensor(self.metrics[threshold]["FP"])

            TP_4x4 = torch.tensor(self.metrics[threshold]["TP_4x4"])
            FN_4x4 = torch.tensor(self.metrics[threshold]["FN_4x4"])
            FP_4x4 = torch.tensor(self.metrics[threshold]["FP_4x4"])

            TP_16x16 = torch.tensor(self.metrics[threshold]["TP_16x16"])
            FN_16x16 = torch.tensor(self.metrics[threshold]["FN_16x16"])
            FP_16x16 = torch.tensor(self.metrics[threshold]["FP_16x16"])

            score_dict[threshold]["csi"] = (TP / (TP + FN + FP)).cpu().tolist()
            score_dict[threshold]["pod"] = (FN / (TP + FN)).cpu().tolist()
            score_dict[threshold]["far"] = (FP / (TP + FP)).cpu().tolist()
            score_dict[threshold]["csi_avg"] = (torch.sum(TP) / (torch.sum(TP) + torch.sum(FN) + torch.sum(FP))).item()
            score_dict[threshold]["pod_avg"] = (torch.sum(FN) / (torch.sum(TP) + torch.sum(FN))).item()
            score_dict[threshold]["far_avg"] = (torch.sum(FP) / (torch.sum(TP) + torch.sum(FP))).item()

            score_dict[threshold]["csi_4x4"] = (TP_4x4 / (TP_4x4 + FN_4x4 + FP_4x4)).cpu().tolist()
            score_dict[threshold]["pod_4x4"] = (FN_4x4 / (TP_4x4 + FN_4x4)).cpu().tolist()
            score_dict[threshold]["far_4x4"] = (FP_4x4 / (TP_4x4 + FP_4x4)).cpu().tolist()
            score_dict[threshold]["csi_4x4_avg"] = (
                torch.sum(TP_4x4) / (torch.sum(TP_4x4) + torch.sum(FN_4x4) + torch.sum(FP_4x4))
            ).item()
            score_dict[threshold]["pod_4x4_avg"] = (torch.sum(FN_4x4) / (torch.sum(TP_4x4) + torch.sum(FN_4x4))).item()
            score_dict[threshold]["far_4x4_avg"] = (torch.sum(FP_4x4) / (torch.sum(TP_4x4) + torch.sum(FP_4x4))).item()

            score_dict[threshold]["csi_16x16"] = (TP_16x16 / (TP_16x16 + FN_16x16 + FP_16x16)).cpu().tolist()
            score_dict[threshold]["pod_16x16"] = (FN_16x16 / (TP_16x16 + FN_16x16)).cpu().tolist()
            score_dict[threshold]["far_16x16"] = (FP_16x16 / (TP_16x16 + FP_16x16)).cpu().tolist()
            score_dict[threshold]["csi_16x16_avg"] = (
                torch.sum(TP_16x16) / (torch.sum(TP_16x16) + torch.sum(FN_16x16) + torch.sum(FP_16x16))
            ).item()
            score_dict[threshold]["pod_16x16_avg"] = (
                torch.sum(FN_16x16) / (torch.sum(TP_16x16) + torch.sum(FN_16x16))
            ).item()
            score_dict[threshold]["far_16x16_avg"] = (
                torch.sum(FP_16x16) / (torch.sum(TP_16x16) + torch.sum(FP_16x16))
            ).item()

        return score_dict


class Evaluator(nn.Module):
    def __init__(self, seq_len, value_scale=80, thresholds=[20, 30, 40]) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.value_scale = value_scale
        self.thresholds = thresholds

        self.ME = ME()
        self.MAE = MAE()
        self.RMSE = RMSE()
        self.CC = CC()
        self.LPIPS = LPIPS()
        self.LPIPS_radar = MyLPIPS()
        self.ME_z = ME()
        self.MAE_z = MAE()
        self.RMSE_z = RMSE()
        self.CC_z = CC()
        self.SSIM = SSIM()
        self.BinaryEvaluator = BinaryEvaluator(seq_len, thresholds)

    def forward(self, pred, gt):
        self.ME(pred, gt)
        self.MAE(pred, gt)
        self.RMSE(pred, gt)
        self.CC(pred, gt)
        self.LPIPS(pred[:, :, 0], gt[:, :, 0])
        self.LPIPS_radar(pred[:, :, 0], gt[:, :, 0])
        self.SSIM(pred[:, :, 0], gt[:, :, 0])
        pred = pred[:, :, 0] * self.value_scale
        gt = gt[:, :, 0] * self.value_scale
        self.ME_z(pred, gt)
        self.MAE_z(pred, gt)
        self.RMSE_z(pred, gt)
        self.CC_z(pred, gt)
        self.BinaryEvaluator(pred, gt)

    def get_metrics(self):
        me = self.ME.calculate()
        me_avg = me.mean().item()
        mae = self.MAE.calculate()
        mae_avg = mae.mean().item()
        rmse = self.RMSE.calculate()
        rmse_avg = rmse.mean().item()
        cc = self.CC.calculate()
        cc_avg = cc.mean().item()
        lpips = self.LPIPS.calculate()
        lpips_avg = lpips.mean().item()
        lpips_radar = self.LPIPS_radar.calculate()
        lpips_radar_avg = lpips_radar.mean().item()
        me_z = self.ME_z.calculate()
        me_z_avg = me_z.mean().item()
        mae_z = self.MAE_z.calculate()
        mae_z_avg = mae_z.mean().item()
        rmse_z = self.RMSE_z.calculate()
        rmse_z_avg = rmse_z.mean().item()
        cc_z = self.CC_z.calculate()
        cc_z_avg = cc_z.mean().item()
        ssim = self.SSIM.calculate()
        ssim_avg = ssim.mean().item()

        metrics = {
            "ME": me.cpu().tolist(),
            "ME_avg": me_avg,
            "MAE": mae.cpu().tolist(),
            "MAE_avg": mae_avg,
            "RMSE": rmse.cpu().tolist(),
            "RMSE_avg": rmse_avg,
            "CC": cc.cpu().tolist(),
            "CC_avg": cc_avg,
            "ME-z": me_z.cpu().tolist(),
            "ME-z_avg": me_z_avg,
            "MAE-z": mae_z.cpu().tolist(),
            "MAE-z_avg": mae_z_avg,
            "RMSE-z": rmse_z.cpu().tolist(),
            "RMSE-z_avg": rmse_z_avg,
            "CC-z": cc_z.cpu().tolist(),
            "CC-z_avg": cc_z_avg,
            "SSIM": ssim.cpu().tolist(),
            "SSIM_avg": ssim_avg,
            "LPIPS": lpips.cpu().tolist(),
            "LPIPS_avg": lpips_avg,
            "LPIPS-radar": lpips_radar.cpu().tolist(),
            "LPIPS-radar_avg": lpips_radar_avg,
        }
        metrics.update(self.BinaryEvaluator.calculate())
        return metrics


if __name__ == "__main__":
    eval = Evaluator(20)

    pred = torch.rand((4, 20, 4, 36, 64, 64))
    gt = torch.rand((4, 20, 4, 36, 64, 64))
    pred = pred.to("cuda")
    gt = gt.to("cuda")

    eval(pred, gt)
    print(eval.get_metrics())
