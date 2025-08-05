import numpy as np
import torch


def compute_ptcloud_from_depth(opt, depth, H, W, focal_length):
    '''
    make a point cloud by unprojecting the pixels with depth
    '''
    XYZ = torch.zeros(H, W, 3, device=opt.device)
    
    # fill assets with background
    H_bg_min = -opt.cam_principal_y
    W_bg_min = -opt.cam_principal_x

    H_bg_max = H - opt.cam_principal_y
    W_bg_max = W - opt.cam_principal_x
    
    H_ = torch.linspace(H_bg_min, H_bg_max, H, device=opt.device)
    W_ = torch.linspace(W_bg_min, W_bg_max, W, device=opt.device)
    r, c = torch.meshgrid(H_, W_, indexing='ij')

    XYZ[:, :, 0] = (depth / focal_length) * c
    XYZ[:, :, 1] = (depth / focal_length) * r
    XYZ[:, :, 2] = depth

    return XYZ


def compute_ptcloud_from_depth_np(opt, depth, H, W, focal_length):
    '''
    make a point cloud by unprojecting the pixels with depth
    '''
    XYZ = np.zeros((H, W, 3), dtype=np.float32)
    
    # fill assets with background
    H_bg_min = -opt.cam_principal_y
    W_bg_min = -opt.cam_principal_x

    H_bg_max = H - opt.cam_principal_y
    W_bg_max = W - opt.cam_principal_x
    
    H_ = np.linspace(H_bg_min, H_bg_max, H)
    W_ = np.linspace(W_bg_min, W_bg_max, W)
    r, c = np.meshgrid(H_, W_, indexing='ij')

    XYZ[:, :, 0] = (depth / focal_length) * c
    XYZ[:, :, 1] = (depth / focal_length) * r
    XYZ[:, :, 2] = depth

    return XYZ


def compute_light_direction(ptcloud, light_pos):
    '''
    ptcloud:    (B, H, W, 3)
    light_pos:  (R, C, 3)
    '''
    
    incident = light_pos.unsqueeze(2).unsqueeze(2).unsqueeze(2) - ptcloud.unsqueeze(0).unsqueeze(0)
    incident = incident / (torch.linalg.norm(incident, axis=-1).unsqueeze(-1) + 1e-8)
    exitant = -(ptcloud / (torch.linalg.norm(ptcloud, axis=-1).unsqueeze(-1) + 1e-8))

    return incident, exitant[None, None, ...]

