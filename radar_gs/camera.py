import torch
from torch import nn
from utils.graphics_utils import getWorld2View


class Camera(nn.Module):
    def __init__(self, 
                 cam_info,
                 ):
        super(Camera, self).__init__()

        self.R = cam_info.R
        self.T = cam_info.T
        self.view_depth = cam_info.view_depth
        self.gt_image = torch.tensor(cam_info.gt_image, dtype=torch.float32)
        self.image_channel = cam_info.channel
        self.image_width = cam_info.width
        self.image_height = cam_info.height

        self.world_view_transform = torch.tensor(getWorld2View(self.R, self.T)).transpose(0, 1)

    
def cameraList_from_camInfos(cam_infos):
    # wrapper function to create a list of Camera objects
    cam_list = []

    for cam_info in cam_infos:
        cam_list.append(Camera(cam_info))

    return cam_list