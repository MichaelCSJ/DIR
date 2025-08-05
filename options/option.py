import argparse
import os
import torch
import numpy as np
import datetime

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
        self.parser.add_argument('--name', type=str, default='train')
        self.parser.add_argument('--num_threads', type=int, default=0)
        self.parser.add_argument('--batch_size', type=int, default=1)
        self.parser.add_argument('--light_N', type=int, default=4)
        self.parser.add_argument('--initial_pattern', type=str, default='mono_gradient')
        self.parser.add_argument('--use_learned_pattern', action='store_true', default=True)
        self.parser.add_argument('--num_basis_BRDFs', type=int, default=7)        
        self.parser.add_argument('--BRDF_fitting_interation', type=int, default=1000)    
         
        self.parser.add_argument('--lr_depth', type=float, default=3e-2) 
        self.parser.add_argument('--lr_coeff_map', type=float, default=3e-2)  
        self.parser.add_argument('--lr_da', type=float, default=1e-1)  
        self.parser.add_argument('--lr_basis_sa', type=float, default=1e-2)  
        self.parser.add_argument('--lr_basis_rg', type=float, default=1e-2)  
        self.parser.add_argument('--lr_normal', type=float, default=1e-1)  
        
        self.parser.add_argument('--fix_light_pattern', action='store_true', default=True)
        self.parser.add_argument('--load_epoch', type=int, default=0) # epoch that previous stoped
        self.parser.add_argument('--load_step_start', type=int, default=0) # previous step
        
        self.parser.add_argument('--dataset_root', type=str, default='/bean/DCdataset/')
        self.parser.add_argument('--backlight_path', type=str, default='./calibration/parameters/backlight.pth')
        self.parser.add_argument('--save_dir', type=str, default='./results/')
        self.parser.add_argument('--light_geometric_calib_fn', type=str, default='./calibration/parameters/finetuned_position.npy')

        self.parser.add_argument('--load_monitor_light', action='store_true', default=False)
        self.parser.add_argument('--light_fn', type=str)

        # camera
        self.parser.add_argument('--cam_R', type=int, default=512) 
        self.parser.add_argument('--cam_C', type=int, default=612)

        self.parser.add_argument('--cam_focal_length', type=float, default=1.1418e3)
        self.parser.add_argument('--cam_pitch', type=float, default=3.45e-6*2) #6e-6
        self.parser.add_argument('--cam_principal_x', type=float, default=3.182451e+02) 
        self.parser.add_argument('--cam_principal_y', type=float, default=2.631557e+02) 


        self.parser.add_argument('--cam_gain', type=float, default=0.0)
        self.parser.add_argument('--cam_gain_base', type=float, default=1.0960) 
        self.parser.add_argument('--static_interval', type=float, default=1/24) # 1s/fps

        # rendering
        self.parser.add_argument('--noise_sigma', type=float, default=0.001) 
        self.parser.add_argument('--rendering_scalar', type=float, default= 1.3e2/40)
        self.parser.add_argument('--GGX_rendering_scalar', type=float, default= 1) 
        self.parser.add_argument('--pattern_rendering_scalar', type=float, default= 1/(24))
        

        # monitor
        self.parser.add_argument('--light_R', type=int, default=9)
        self.parser.add_argument('--light_C', type=int, default=16)
        self.parser.add_argument('--light_pitch', type=float, default=0.315e-3 * 120 * 2) 
        self.parser.add_argument('--monitor_gamma', type=float, default=2.2) 

        self.initialized = True


    def parse(self, save=True, isTrain=True, parse_args=None):
        if not self.initialized:
            self.initialize()
        if parse_args is None:
            self.opt = self.parser.parse_args()
            self.opt = self.parser.parse_args()
        else:
            self.opt = self.parser.parse_args(parse_args)

        self.opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.opt.dtype = torch.float32

        self.opt.cam_focal_length_mm = self.opt.cam_focal_length * self.opt.cam_pitch
        self.opt.cam_shutter_time = self.opt.static_interval / self.opt.light_N

        self.opt.original_R = self.opt.cam_R
        self.opt.original_C = self.opt.cam_C
        
        # fall_off coefficients
        self.opt.coeff = [-5.4580, 14.7449]
        
        
        self.opt.dataset_train_root = os.path.join(self.opt.dataset_root, 'train')
        self.opt.dataset_test_root = os.path.join(self.opt.dataset_root, 'val')
        self.opt.tb_dir = os.path.join(self.opt.save_dir, self.opt.name, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.opt.tb_dir, exist_ok=True)
        os.makedirs(os.path.join(self.opt.tb_dir, 'Fitted_parameters'), exist_ok=True)


        # print('------------ Options -------------')
        args = vars(self.opt)
        # print('-------------- End ----------------')

        if save:
            file_name = os.path.join(self.opt.tb_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')

        # Light position
        light_pos = np.load(self.opt.light_geometric_calib_fn)
        light_pos = np.reshape(light_pos, (self.opt.light_R, self.opt.light_C, 3))
        self.opt.light_pos = torch.tensor(light_pos, dtype=self.opt.dtype)
        self.opt.light_pos_np = light_pos

        # backlight: (R, C, 3)
        backlight = torch.load(self.opt.backlight_path)
        self.opt.backlight = torch.sigmoid(backlight.clone().detach().to(dtype=self.opt.dtype))

        if self.opt.use_learned_pattern:
            self.opt.illums = torch.load(f'./patterns/optimized_{self.opt.initial_pattern}.pth')
            
        return self.opt
