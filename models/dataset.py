import torch
import cv2
import numpy as np
import os
from models.utils import compute_ptcloud_from_depth_np

class StereoDataset(torch.utils.data.Dataset):
    def __init__(self, opt, list_name):
        self.opt = opt
        self.scene_names = file_load(os.path.join(opt.dataset_root, list_name))
        
    def __getitem__(self, i):
        scene_name = self.scene_names[i]
        basis_main_dir = os.path.join(self.opt.dataset_root, scene_name, 'main/diffuseNspecular')
        
        mask = cv2.imread(os.path.join(self.opt.dataset_root, scene_name, 'mask.png'))/255
        mask[mask!=0] = 1
        mask = mask[:,:,0]
                
        # kernel = np.ones((3,3), np.uint8)
        # mask = cv2.erode(mask, kernel, iterations=1)
        
        ptcloud = np.load(os.path.join(self.opt.dataset_root, scene_name, 'point_cloud.npy'))
        depth = ptcloud[:,:,-1]
        
        masked_depth = depth*mask
        background = masked_depth==0
        masked_depth[background] = 2
        
        
        ptcloud = compute_ptcloud_from_depth_np(self.opt, masked_depth, self.opt.cam_R, self.opt.cam_C, self.opt.cam_focal_length)    
        
        basis_file_names = [str(f).zfill(3)+'.png' for f in range(144)]
        basis_main = np.zeros((self.opt.light_R*self.opt.light_C, self.opt.cam_R, self.opt.cam_C, 3), dtype=np.float32)
        for idx, basis_file in enumerate(basis_file_names):
            hdr_main = cv2.imread(os.path.join(basis_main_dir, basis_file))/255.
            hdr_main = np.clip(hdr_main[:,:,::-1], 0, None)
            basis_main[idx] = hdr_main
            
        
        input_dict = {
            'id': i,
            'scene': scene_name,
            'OLAT_main': basis_main,
            'depth': masked_depth.astype(np.float32),
            'ptcloud': ptcloud,
            'mask': mask.astype(np.float32)
        }
        
        return input_dict
            
    def __len__(self):
        return len(self.scene_names)
    
        
        
def file_load(path):
    # Read data list
    data_path = []
    f = open("{0}.txt".format(path), 'r')
    while True:
        line = f.readline()
        if not line:
            break
        data_path.append(line[:-1])
    f.close()
    return data_path

