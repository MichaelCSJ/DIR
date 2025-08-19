import os
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import torch
from torch.nn import functional as F
import scipy.io as sio
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import warnings

from models.renderer import *
from models.reconstructor import *
from utils.utils import compute_light_direction, compute_ptcloud_from_depth

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

class Trainer():
    def __init__(self, opt, renderer, reconstructor, writer, last_epoch):
        self.opt = opt
        self.renderer = renderer
        self.reconstructor = reconstructor
        self.writer = writer
        self.device = self.opt.device
        self.total_step = 1
        self.softmax = torch.nn.Softmax(dim=1)

        # Load display pattern 
        self.display_pattern = self.opt.illums.clone().detach()
            
        # Optimization variable parameterization
        self.monitor_OLAT_pos = self.opt.light_pos.clone().detach().to(self.device)
        self.camera_gain = torch.logit(torch.tensor([self.opt.cam_gain], dtype=self.opt.dtype)/48).to(self.device)
        self.backlight = self.opt.backlight.clone().detach().to(self.device)
        
    def segment_color_hsv_chromaticity(self, albedo_est, mask):
        (B, H, W, N, R, C, num_basis_BRDFs) = self.dimensions
        
        # Initialize output tensors
        palettes = torch.zeros(B, num_basis_BRDFs, 3).to(self.device)  # 3 for RGB
        weights = torch.zeros(B, num_basis_BRDFs, H, W).to(self.device)
        for b in range(B):
            target_img = albedo_est[b].detach().cpu().numpy()
            mask_img = mask[b].detach().cpu().numpy()
            
            # Convert RGB to HSV
            hsv_img = matplotlib.colors.rgb_to_hsv(target_img)
            Hue, S, V = hsv_img[..., 0], hsv_img[..., 1], hsv_img[..., 2]

            # Use only H and S for chromaticity
            chromaticity_img = np.stack((Hue, S), axis=-1)

            # Apply mask to include only object pixels
            valid_pixels = mask_img > 0  # Mask: 1 for objects, 0 for background
            vec_img = chromaticity_img[valid_pixels]  # Extract only valid pixels

            # Perform k-means clustering
            model = KMeans(n_clusters=num_basis_BRDFs)
            pred = model.fit_predict(vec_img)
            cluster_centers = model.cluster_centers_  # Cluster centers in chromaticity space (H, S)

            # Convert cluster centers back to RGB via HSV
            reconstructed_hsv = np.zeros((num_basis_BRDFs, 3))
            reconstructed_hsv[:, 0] = cluster_centers[:, 0]  # H
            reconstructed_hsv[:, 1] = cluster_centers[:, 1]  # S
            reconstructed_hsv[:, 2] = 1.0  # V (max intensity)

            reconstructed_rgb = matplotlib.colors.hsv_to_rgb(reconstructed_hsv)
            palette = torch.tensor(reconstructed_rgb).to(self.device)
            palettes[b] = palette

            # Generate weight maps
            labels = np.full((H, W), -1)  # Initialize labels with -1 (background)
            labels[valid_pixels] = pred  # Assign cluster labels to valid pixels

            one_hot_encoder = OneHotEncoder(categories='auto', sparse_output=False)
            labels_one_hot = one_hot_encoder.fit_transform(labels[valid_pixels].reshape(-1, 1))
            weight = torch.zeros(H, W, num_basis_BRDFs).to(self.device)  # Initialize weights with 0
            weight[valid_pixels] = torch.tensor(labels_one_hot).to(self.device).to(self.opt.dtype)
            weight = weight.permute(2, 0, 1)
            weights[b] = weight

        return weights, palettes
    
    def sampling_roughness(self):
        (B, H, W, N, R, C, num_basis_BRDFs) = self.dimensions
        
        # data = [0.02, 0.05, 0.13, 0.34]
        # kde = gaussian_kde(data, bw_method='scott')
        # num_samples = B*num_basis_BRDFs
        # samples = kde.resample(num_samples)[0]
        # samples_clipped = np.clip(samples, self.opt.roughness_min, self.opt.roughness_max)
        # samples_normalized = np.clip(samples_clipped, 1e-8, 1-1e-8)
        # basis_rg = torch.tensor(samples_normalized, device=self.opt.device, requires_grad=True)
        
        basis_rg = torch.ones(B, num_basis_BRDFs, device=self.device, requires_grad=True) * 0.5
        return basis_rg.reshape(B, num_basis_BRDFs, 1, 1, 1).to(dtype=torch.float32)
    
    def init_BRDF_CT(self, initial_depth, initial_normal, initial_diffuse, weights, pallete, mask):
        (B, H, W, N, R, C, num_basis_BRDFs) = self.dimensions
        
        # Optimizable parameters
        basis_da = pallete.reshape(B, num_basis_BRDFs, 3, 1, 1)*0.8+0.1
        basis_sa = torch.ones(B, num_basis_BRDFs, 3, 1, 1, device=self.device)/2
        basis_rg = self.sampling_roughness()
        
        self.depth = torch.nn.Parameter(initial_depth)
        self.diffuse_albedo = torch.nn.Parameter(torch.logit(initial_diffuse.permute(0,3,1,2)))        
        self.coeff_map = torch.nn.Parameter((weights*10+1))
        self.basis_da = torch.nn.Parameter(torch.logit(basis_da))
        self.basis_sa = torch.nn.Parameter(torch.logit(basis_sa))
        self.basis_rg = torch.nn.Parameter(torch.logit(basis_rg))
        self.normal = torch.nn.Parameter(initial_normal)

        self.optimizer_BRDF = torch.optim.AdamW([
            {'params': self.coeff_map, 'lr': self.opt.lr_coeff_map},
            {'params': self.basis_da, 'lr': self.opt.lr_da},
            {'params': self.basis_sa, 'lr': self.opt.lr_basis_sa},
            {'params': self.basis_rg, 'lr': self.opt.lr_basis_rg},
            {'params': self.normal, 'lr': self.opt.lr_normal}
        ])

        return 
    
    def reconstruct_one_step(self, mask, OLAT_main):
        (B, H, W, N, R, C, num_basis_BRDFs) = self.dimensions
        
        # Render image
        OLAT_original_gt = torch.clamp(OLAT_main, 1e-8, 1)
        I_main_gt = self.renderer.render(OLAT_main, self.display_pattern, self.camera_gain)
        I_main_gt = torch.clamp(I_main_gt, 1e-8, 1)
        
        for b in range(B):
            valid_pixels = torch.nonzero(mask[b])
            permuted_indices = torch.randperm(valid_pixels.size(0))
            sampled_coords = valid_pixels[permuted_indices]
            sampled_num = sampled_coords.size(0)
            
            depth = self.depth[b]
            coeff_map    = self.softmax(self.coeff_map)
            basis_da     = torch.sigmoid(self.basis_da).repeat(1,1,1,1,sampled_num)
            basis_rg     = torch.sigmoid(self.basis_rg).repeat(1,1,1,1,sampled_num)#/2+0.01
            basis_sa     = torch.sigmoid(self.basis_sa).repeat(1,1,1,1,sampled_num)
            normal = F.normalize(self.normal, p=2, dim=-1)
            ptcloud = compute_ptcloud_from_depth(self.opt, depth, self.opt.cam_R, self.opt.cam_C, self.opt.cam_focal_length)    
            
            coeff_sampled = coeff_map[b,:,sampled_coords[:, 0],sampled_coords[:, 1]]
            n_sampled = normal[b, sampled_coords[:, 0], sampled_coords[:, 1]].reshape(B,1,sampled_num,3).permute(0,3,1,2)
            points_batch = ptcloud[sampled_coords[:, 0], sampled_coords[:, 1]]
        
        OLAT_losses = 0
        rendering_losses = 0
        for i in range(0, 1):
            incident, exitant = compute_light_direction(points_batch.reshape(1,1,sampled_num,3), self.monitor_OLAT_pos)
            exitant = exitant[0,0].permute(0, 3, 1, 2)
            distance = torch.sqrt(torch.sum(torch.pow(self.monitor_OLAT_pos.unsqueeze(2).unsqueeze(2).unsqueeze(2) - points_batch.reshape(1,1,sampled_num,3).unsqueeze(0).unsqueeze(0), 2), axis=-1))
            falloff_sampled = 1/(self.opt.coeff[0]+self.opt.coeff[1]*(distance)**2)
            _, _, OLAT_est, nl, nh = self.renderer.render_Cook_Torrance(n_sampled, incident, exitant, basis_da, basis_sa, basis_rg, self.camera_gain, coeff_sampled, falloff_sampled, self.backlight)
                
            rendered_est = self.renderer.render(OLAT_est, self.display_pattern, self.camera_gain)
            OLAT_est = torch.clamp(OLAT_est,1e-8,1)
            I_sampled = I_main_gt[:,:,sampled_coords[:, 0], sampled_coords[:, 1]].unsqueeze(2)
            OLAT_sampled = OLAT_original_gt[:,:,sampled_coords[:, 0], sampled_coords[:, 1]].unsqueeze(2)
            
            OLAT_est_gamma = OLAT_est[:] ** (1/2.2)
            OLAT_sampled_gamma = OLAT_sampled[:] ** (1/2.2)
            rendered_est_gamma = rendered_est ** (1/2.2)
            I_sampled_gamma = I_sampled ** (1/2.2)
            
            
            def total_variation_loss(img):
                img = img*mask[...,None]
                tv_h = torch.sqrt(torch.mean((torch.abs(img[:, 1:, :, :] - img[:, :-1, :, :])**2)))
                tv_w = torch.sqrt(torch.mean((torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])**2)))
                return tv_h + tv_w
            
            OLAT_loss = torch.sqrt(torch.mean(((OLAT_est_gamma - OLAT_sampled_gamma)*255)**2))
            rendering_loss = torch.sqrt(torch.mean(((rendered_est_gamma - I_sampled_gamma)*255)**2))
            
            tv_weight = 20
            tv_term = total_variation_loss((-normal+1)/2)
            if self.opt.use_multiplexing:
                loss_with_tv = rendering_loss + tv_weight * tv_term
            else:
                loss_with_tv = OLAT_loss + tv_weight * tv_term
            
            loss = (loss_with_tv)
            loss.backward()
                        
            OLAT_losses += (OLAT_loss)*sampled_num
            rendering_losses += (rendering_loss)*sampled_num 
            
        self.optimizer_BRDF.step()
        self.optimizer_BRDF.zero_grad()
        #================================================================================================
        
        loss_dict = {}
        loss_dict['OLAT'] = OLAT_losses/torch.sum(mask)
        loss_dict['Rendering'] = rendering_losses/torch.sum(mask)
        
        self.writer.add_scalar('Scene: Total loss', sum(loss_dict.values()), self.total_step)
        self.writer.add_scalar('Scene: OLAT loss', loss_dict['OLAT'], self.total_step)
        self.writer.add_scalar('Scene: Rendering loss', loss_dict['Rendering'], self.total_step)
        
        return loss_dict
          
    def run_model(self, data):
        file_names = data['scene'][0]
        B = self.opt.batch_size
        H = self.opt.cam_R
        W = self.opt.cam_C
        N = self.opt.light_N
        (R, C) = (self.opt.light_R, self.opt.light_C)
        num_basis_BRDFs = self.opt.num_basis_BRDFs
        # TODO: Metadata: (+)mask, num_poi
        self.dimensions = (B, H, W, N, R, C, num_basis_BRDFs) 
        
        OLAT_main = data['OLAT_main'].to(device=self.device)
        depth = data['depth'].to(device=self.device)
        ptcloud = data['ptcloud'].to(device=self.device)
        mask = data['mask'].to(device=self.device)
        num_poi = data['mask'].sum()
        print("valid pixel number: ", num_poi)
        
        '''
        wi: (R,C,B,H,W,xyz)
        wo: (B,H,W,xyz)
        falloff: (R,C,B,H,W)
        L_pos: (R,C,xyz)
        pts: (B,H,W,xyz)
        weights: (B,basis,H,W), ont-hot encoded
        palette: (B,basis,rgb)
        mask: (B,H,W)
        basis_da: (B,basis,3,1,1)
        basis_sa: (B,basis,3,1,1)
        basis_rg: (B,basis,1,1,1)
        '''
        with torch.no_grad():
            # Light direction
            incident, exitant = compute_light_direction(ptcloud, self.monitor_OLAT_pos)
            exitant = exitant[0,0].permute(0, 3, 1, 2)
            distance = torch.sqrt(torch.sum(torch.pow(self.monitor_OLAT_pos.unsqueeze(2).unsqueeze(2).unsqueeze(2) - ptcloud.unsqueeze(0).unsqueeze(0), 2), axis=-1))
            falloff = 1/(self.opt.coeff[0]+self.opt.coeff[1]*(distance)**2)
            incident = incident * falloff[...,None]

            I_diffuse_gt = self.renderer.render(OLAT_main, self.display_pattern, self.camera_gain)
                
            # Reconstruct normal and albedo
            recon_output = self.reconstructor.forward(I_diffuse_gt, self.display_pattern, incident, self.camera_gain)
            normal_est, albedo_est = (recon_output['normal'], recon_output['albedo'])
            weights, palette = self.segment_color_hsv_chromaticity(albedo_est.detach().clone(), mask)
            self.init_BRDF_CT(depth, normal_est, albedo_est.detach().clone(), weights, palette, mask)
            
            self.save_dir = os.path.join(self.opt.tb_dir, file_names)
            os.makedirs(self.save_dir, exist_ok=True)
            mask_np = mask[0][...,None].detach().cpu().numpy()
            for i in range(self.opt.light_N):
                I_save = (I_diffuse_gt[0,i]).detach().cpu().numpy()
                I_save = (I_save*255).astype(np.uint8)
                save_path = os.path.join(self.save_dir, f'rendered_{i:02d}th.png')
                cv2.imwrite(save_path, I_save[:,:,::-1])
            normal_vis = (-normal_est[0]).detach().cpu().numpy()
            normal_vis[:,:,0] = -normal_vis[:,:,0]
            normal_vis = (normal_vis+1)/2
            normal_save = (normal_vis*255).astype(np.uint8)*mask_np
            save_path = os.path.join(self.save_dir, f'initial_normal.png')
            cv2.imwrite(save_path, normal_save[:,:,::-1])
            
        for iter in tqdm(range(self.opt.BRDF_fitting_interation), desc=f"Inverse rendering: {file_names}", leave=False):                
            with torch.no_grad():
                if iter==(self.opt.BRDF_fitting_interation-1):
                    self.save_N_display(iter, mask, OLAT_main)
            loss_dict = self.reconstruct_one_step(mask, OLAT_main)            
            self.total_step += 1
            
        model_results = {}
        model_results['OLAT'] = loss_dict['OLAT']
        return model_results['OLAT'].item()
      
          
    def save_N_display(self, iter, mask, OLAT_main):
        (B, H, W, N, R, C, num_basis_BRDFs) = self.dimensions
        device = self.device
        iter_str = str(iter).zfill(5)
        
        def depth_to_normal(ptcloud):
            output = torch.zeros_like(ptcloud)
            dx = torch.cat([ptcloud[2:, 1:-1] - ptcloud[:-2, 1:-1]], dim=0)
            dy = torch.cat([ptcloud[1:-1, 2:] - ptcloud[1:-1, :-2]], dim=1)
            normal_map = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1, eps=1e-6)
            output[1:-1, 1:-1, :] = normal_map
            return output
        
        def gamma(img):
            return img**(1/2.2)
        
        #================================================================================================
        # Save parameters  
        log_dir =os.path.join(self.save_dir,'parameters')
        os.makedirs(log_dir, exist_ok=True)
        torch.save(self.display_pattern, os.path.join(log_dir, f'{iter_str}_patterns.pth'))
        torch.save(self.depth.detach(), os.path.join(log_dir, f'{iter_str}_depth.pth'))
        torch.save(self.coeff_map.detach(), os.path.join(log_dir, f'{iter_str}_weight_map.pth'))
        torch.save(self.basis_da.detach(), os.path.join(log_dir, f'{iter_str}_diffuse_albedo.pth'))
        torch.save(self.basis_sa.detach(), os.path.join(log_dir, f'{iter_str}_specular_albedo.pth'))
        torch.save(self.basis_rg.detach(), os.path.join(log_dir, f'{iter_str}_specular_roughness.pth'))
        torch.save(self.normal.detach(), os.path.join(log_dir, f'{iter_str}_normal.pth'))
        torch.save(mask.detach(), os.path.join(log_dir, f'{iter_str}_mask.pth'))
        #==========================================================================================
        
        #================================================================================================
        # Render image
        I_main_gt = self.renderer.render(OLAT_main, self.display_pattern, self.camera_gain)
        I_main_gt = torch.clamp(I_main_gt, 0, 1)
                
        # Pixel sampling
        valid_pixels = torch.nonzero(mask[0])
        permuted_indices = torch.randperm(valid_pixels.size(0))
        valid_pixels = valid_pixels[permuted_indices]
        sampled_coords = valid_pixels
        sampled_num = sampled_coords.size(0)
        
        OLAT_result = torch.zeros(144,H,W,3, device=device)
        Rendered_result = torch.zeros(N,H,W,3, device=device)
        Rendered_fd = torch.zeros(N,H,W,3, device=device)
        Rendered_fs = torch.zeros(N,H,W,3, device=device)
        nl_map = torch.zeros(144,H,W, device=device)
        shadow_map = torch.zeros(144,H,W, device=device)
        nh_map = torch.zeros(H,W, device=device)
        
        depth = self.depth[0]
        coeff_map    = self.softmax(self.coeff_map)
        basis_da     = torch.sigmoid(self.basis_da)
        basis_rg     = torch.sigmoid(self.basis_rg)
        basis_sa     = torch.sigmoid(self.basis_sa)
        normal = F.normalize(self.normal, p=2, dim=-1)
        ptcloud = compute_ptcloud_from_depth(self.opt, depth, self.opt.cam_R, self.opt.cam_C, self.opt.cam_focal_length)    
        
        points_batch = ptcloud[sampled_coords[:, 0], sampled_coords[:, 1]]
        n_sampled = normal[:, sampled_coords[:, 0], sampled_coords[:, 1]].reshape(B,1,sampled_num,3).permute(0,3,1,2)
        coeff_sampled = coeff_map[:,:,sampled_coords[:, 0],sampled_coords[:, 1]]
        
        incident, exitant = compute_light_direction(points_batch.reshape(1,1,sampled_num,3), self.monitor_OLAT_pos)
        exitant = exitant[0,0].permute(0, 3, 1, 2)
        distance = torch.sqrt(torch.sum(torch.pow(self.monitor_OLAT_pos.unsqueeze(2).unsqueeze(2).unsqueeze(2) - points_batch.reshape(1,1,sampled_num,3).unsqueeze(0).unsqueeze(0), 2), axis=-1))
        falloff_sampled = 1/(self.opt.coeff[0]+self.opt.coeff[1]*(distance)**2)
        OLAT_fd, OLAT_fs, OLAT_est, nl, nh = self.renderer.render_Cook_Torrance(n_sampled, incident, exitant, basis_da, basis_sa, basis_rg, self.camera_gain, coeff_sampled, falloff_sampled, self.backlight)
            
        rendered_est = self.renderer.render(OLAT_est, self.display_pattern, self.camera_gain)
        fd = self.renderer.render(OLAT_fd, self.display_pattern, self.camera_gain)
        fs = self.renderer.render(OLAT_fs, self.display_pattern, self.camera_gain)
        
        
        OLAT_result[:,sampled_coords[:, 0],sampled_coords[:, 1]] = OLAT_est[0,:,0].clamp(min=0,max=1)
        Rendered_result[:,sampled_coords[:, 0],sampled_coords[:, 1]] = rendered_est[0,:,0].clamp(min=0,max=1)
        Rendered_fd[:,sampled_coords[:, 0],sampled_coords[:, 1]] = fd[0,:,0].clamp(min=0,max=1)
        Rendered_fs[:,sampled_coords[:, 0],sampled_coords[:, 1]] = fs[0,:,0].clamp(min=0,max=1)
        nh_map[sampled_coords[:, 0],sampled_coords[:, 1]] = torch.clamp(nh[0,0,0].sum(0)/(R*C), 0, 1)
        nl_map[:,sampled_coords[:, 0],sampled_coords[:, 1]] = nl[0]

        OLAT_file_names = [str(f).zfill(3)+'.png' for f in range(144)]
        for i in range(len(OLAT_file_names)):
            save_path = os.path.join(self.save_dir, OLAT_file_names[i])
            save_img = OLAT_result[i].detach().cpu().numpy()[:,:,::-1]
            cv2.imwrite(save_path, (save_img*255).astype(np.uint8))
        
        normal_save = (normal[0]*mask[0][...,None]).detach().cpu().numpy()
        normal_save[:,:,0] = -normal_save[:,:,0]
        normal_save = (-normal_save+1)/2
        cv2.imwrite(os.path.join(self.save_dir, 'optimized_normal.png'), (normal_save[:,:,::-1]*255).astype(np.uint8))
        
        
        normal_from_depth = depth_to_normal(ptcloud)
        cosine_similarity = 1 - mask[0]*F.cosine_similarity(normal[0], normal_from_depth, dim=-1)
        
        
        #================================================================================================
        # 0.Patterns  
        plt.figure(figsize=(3*self.opt.light_N,3))
        plt.suptitle(f'{iter}th Learned Patterns')
        for b in range(self.opt.light_N):
            plt.subplot(1,self.opt.light_N,b+1)
            plt.title(f'Pattern {b}')
            plt.imshow(torch.sigmoid(self.display_pattern[b]).detach().cpu().numpy(), vmax=1)
            plt.axis('off')
        fig = plt.gcf()
        canvas = FigureCanvas(fig)
        canvas.draw()
        
        learned_patterns = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        learned_patterns = learned_patterns.reshape(fig.canvas.get_width_height()[::-1]+(3,))
        self.writer.add_image(f'0. Pattern', torch.tensor(learned_patterns).permute(2,0,1), self.total_step)
        plt.close(fig)
        torch.cuda.empty_cache()
        #================================================================================================
        # 1.Reflectance  
        path = './calibration/visualization'
        sphere_normal = sio.loadmat(os.path.join(path, 'sphere_normal.mat'))
        sphere_normal = sphere_normal['Normal_gt'].astype(np.float32)
        sphere_mask = cv2.imread(os.path.join(path, 'sphere_mask.png'))/255
        sphere_mask[sphere_mask!=0] = 1
        sphere_mask = torch.tensor(sphere_mask[:,:,0])
        sphere_mask = sphere_mask.to(device=device)

        sphere_n = torch.tensor(sphere_normal).permute(2,0,1).to(device)
        view = torch.zeros_like(sphere_n).to(device)
        view[2, ...] = 1
        light1 = view
        light2 = torch.ones_like(sphere_n).to(device)
        light3 = -(torch.ones_like(sphere_n).to(device))
        light3[2, ...] = -3
        light1 = F.normalize(light1, dim=0)
        light2 = F.normalize(light2, dim=0)
        light3 = F.normalize(light3, dim=0)
        light = torch.concat([light3[:,None,...], light2[:,None,...]], dim=1)
        da = torch.sigmoid(self.basis_da)**(1/2.2)
        specular_albedo = torch.sigmoid(self.basis_sa)
        specular_roughness = torch.sigmoid(self.basis_rg)#/2+0.01
                
        plt.figure(figsize=(3*self.opt.num_basis_BRDFs,6))
        plt.suptitle(f'{iter}th Basis BRDF')
        for b in range(self.opt.num_basis_BRDFs):
            plt.subplot(2,self.opt.num_basis_BRDFs,b+1)
            plt.title(f'rho_s: {specular_albedo[0,b,0,0,0]:.2f}, {specular_albedo[0,b,1,0,0]:.2f}, {specular_albedo[0,b,2,0,0]:.2f},\n rg: {specular_roughness[0,b,0,0,0]:.2f}')
            nl, fd, fs, rendered = self.renderer.render_Cook_Torrance_sphere(sphere_n, light, view, da[0,b,:,0,0], specular_albedo[0,b,0,0,0], specular_roughness[0,b,0,0,0])
            fs_2d = (rendered).reshape(3,512,612).permute(1,2,0)[150:-150,200:-200]
            fs_2d[sphere_mask[150:-150,200:-200]!=1] = 0.7
            im = plt.imshow(fs_2d.detach().cpu().numpy(), vmax=1, cmap="gray")
            plt.axis('off')
            plt.colorbar()
            
            plt.subplot(2,self.opt.num_basis_BRDFs,b+1+self.opt.num_basis_BRDFs)
            plt.title(f'rho_d: ({da[0,b,0,0,0]:.2f}, {da[0,b,1,0,0]:.2f}, {da[0,b,2,0,0]:.2f})')
            plt.imshow((mask[0]*self.softmax(self.coeff_map)[0,b]).detach().cpu().numpy(), vmin=0, vmax=1)
            plt.axis('off')
            plt.colorbar()
        fig = plt.gcf()
        canvas = FigureCanvas(fig)
        canvas.draw()
        
        basis_spheres = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        basis_spheres = basis_spheres.reshape(fig.canvas.get_width_height()[::-1]+(3,))
        self.writer.add_image(f'1. Basis_BRDFs', torch.tensor(basis_spheres).permute(2,0,1), self.total_step)
        plt.close(fig)
        del nl, fd, fs, rendered
        torch.cuda.empty_cache()
        
        #==========================================================================================
        # 2.Geometry       
        normal = F.normalize(self.normal, p=2, dim=-1)
        depth = self.depth[0]
        plt.figure(figsize=(24,12))
        plt.subplot(4,3,1)
        plt.title('Depth map')
        plt.imshow(depth.detach().cpu().numpy(), vmin=0.4, vmax=0.6)
        plt.colorbar()
        plt.subplot(4,3,2)
        plt.title('Diffuse albedo')
        plt.imshow(torch.sigmoid(self.diffuse_albedo[0].permute(1,2,0)).detach().cpu().numpy())
        plt.subplot(4,3,3)
        plt.title('Theta_h**10')
        plt.imshow((nh_map**10).detach().cpu().numpy(), cmap='hot')
        plt.colorbar()
        
        plt.subplot(4,3,1+3)
        plt.title('Normal from depth')
        plt.imshow((((-normal_from_depth+1)/2)*mask[0].unsqueeze(-1)).detach().cpu().numpy(), vmin=0, vmax=1)
        plt.colorbar()
        plt.subplot(4,3,2+3)
        plt.title('Surface normal')
        plt.imshow((((-normal[0]+1)/2)*mask[0].unsqueeze(-1)).detach().cpu().numpy(), vmin=0, vmax=1)
        plt.subplot(4,3,3+3)
        plt.title('Cosine similarity')
        plt.imshow((cosine_similarity*mask[0]).detach().cpu().numpy(), vmin=0, vmax=1)
        plt.colorbar()
        
        plt.subplot(4,3,1+3+3)
        plt.title('Cast shadow (0th_light)')
        plt.imshow(shadow_map[0].detach().cpu().numpy(), vmin=0, vmax=1)
        plt.subplot(4,3,2+3+3)
        plt.title('Self shadow (0th_light)')
        plt.imshow(nl_map[0].detach().cpu().numpy(), vmin=0, vmax=1)
        plt.subplot(4,3,3+3+3)
        plt.title('Whole shadow (0th_light)')
        plt.imshow((shadow_map[0]*nl_map[0]).detach().cpu().numpy(), vmin=0, vmax=1)
        
        plt.subplot(4,3,1+3+3+3)
        plt.title('Cast shadow (avg)')
        plt.imshow(torch.mean(shadow_map,dim=0).detach().cpu().numpy(), vmin=0, vmax=1)
        plt.subplot(4,3,2+3+3+3)
        plt.title('Self shadow (avg)')
        plt.imshow(torch.mean(nl_map,dim=0).detach().cpu().numpy(), vmin=0, vmax=1)
        plt.subplot(4,3,3+3+3+3)
        plt.title('Whole shadow (avg)')
        plt.imshow((torch.mean(shadow_map,dim=0)*torch.mean(nl_map,dim=0)).detach().cpu().numpy(), vmin=0, vmax=1)
        
        fig = plt.gcf()
        canvas = FigureCanvas(fig)
        canvas.draw()
        Geometry = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        Geometry = Geometry.reshape(fig.canvas.get_width_height()[::-1]+(3,))
        self.writer.add_image(f'2. Geometry', torch.tensor(Geometry).permute(2,0,1), self.total_step)
        plt.close(fig)
        
        #================================================================================================
        # 3.OLAT visulaization
        OLAT_main = torch.clamp(OLAT_main,min=0,max=1)
        plt.figure(figsize=(28,14))
        plt.suptitle('Rendered with display patterns')
        for r in range(3):
            for c in range(4):
                plt.subplot(6,4,(r*8)+4-c)
                plt.imshow(gamma(OLAT_result[64*r+5*c].detach().cpu().numpy()), vmin=0, vmax=1)
                plt.colorbar()
                plt.subplot(6,4,(r*8)+8-c)
                plt.imshow(gamma(OLAT_main[0,64*r+5*c].detach().cpu().numpy()), vmin=0, vmax=1)
                plt.colorbar()
        fig = plt.gcf()
        canvas = FigureCanvas(fig)
        canvas.draw()
        OLAT = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        OLAT = OLAT.reshape(fig.canvas.get_width_height()[::-1]+(3,))
        self.writer.add_image(f'3. OLAT', torch.tensor(OLAT).permute(2,0,1), self.total_step)
        plt.close(fig)
        
        #================================================================================================
        # 4.Rendering results
        plt.figure(figsize=(20,14))
        plt.suptitle('Rendered with display patterns')
        for n in range(self.opt.light_N):
            plt.subplot(3,4,n+1)
            plt.imshow(gamma(Rendered_result[n].detach().cpu().numpy()))
            plt.colorbar()
            plt.subplot(3,4,n+5)
            plt.imshow(gamma(Rendered_fd[n].detach().cpu().numpy()))
            plt.colorbar()
            plt.subplot(3,4,n+9)
            plt.imshow(gamma(Rendered_fs[n].detach().cpu().numpy()))
            plt.colorbar()
        fig = plt.gcf()
        canvas = FigureCanvas(fig)
        canvas.draw()
        Rendered_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        Rendered_img = Rendered_img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
        self.writer.add_image(f'4. Rendered_img', torch.tensor(Rendered_img).permute(2,0,1), self.total_step)
        plt.close(fig)
        
        #================================================================================================
        # 5.Comparison
        plt.figure(figsize=(20,14))
        plt.suptitle('Scene under display patterns')
        for n in range(self.opt.light_N):
            plt.subplot(3,4,n+1)
            plt.title('Capture simulation')
            plt.imshow(gamma(mask[0][...,None]*I_main_gt[0,n]).detach().cpu().numpy())
            plt.colorbar()
            plt.subplot(3,4,n+5)
            plt.title('PBR')
            plt.imshow(gamma(Rendered_result[n].detach().cpu().numpy()))
            plt.colorbar()
            plt.subplot(3,4,n+9)
            plt.title('Error maps')
            plt.imshow(((mask[0][...,None]*torch.clamp(torch.abs(Rendered_result[n] - (I_main_gt)[0,n]),0,1)).mean(dim=-1).detach().cpu().numpy()), cmap='hot', vmin=0, vmax=0.5)
            plt.colorbar()
        fig = plt.gcf()
        canvas = FigureCanvas(fig)
        canvas.draw()
        Comparison = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        Comparison = Comparison.reshape(fig.canvas.get_width_height()[::-1]+(3,))
        self.writer.add_image(f'5. Comparison', torch.tensor(Comparison).permute(2,0,1), self.total_step)
        plt.close(fig)

        return
    