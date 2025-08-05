import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from scipy import ndimage


# =============== HDR weight function =================================
invalid_intensity_ratio = 0.1  # 0 < invalid_intensity_ratio: invalid,  out of 1
# invalid_intensity_ratio = 0.004  # 0 < invalid_intensity_ratio: invalid,  out of 1
max_intenisty = 2**8
weight_trapezoid = np.zeros(max_intenisty)
intv = float(max_intenisty) * invalid_intensity_ratio

for i in range(max_intenisty):
    if i < intv:
        weight_trapezoid[i] = 0  
    elif i < intv * 2:
        weight_trapezoid[i] = (i - intv) / intv
    elif i < max_intenisty - (intv * 2):
        weight_trapezoid[i] = 1
    elif i < max_intenisty - intv:
        weight_trapezoid[i] = (max_intenisty - intv - i) / intv
    else:
        weight_trapezoid[i] = 0
        

def make_hdr(ldr_images, exposure):
    # ex_time = np.array([160, 200, 240, 270, 300, 340])
    # ex_max = 340
    # exposure = [x / ex for x in exposure] 
    exposure = [x / min(exposure) for x in exposure] 

    # ldr_images_filtered = [ndimage.median_filter(image, size=3) for image in ldr_images]
    weighted_images = [weight_trapezoid[image] for image in ldr_images]
    radiance_images = [np.multiply(weighted_images[i], ldr_images[i] / exposure[i]) for i in range(len(ldr_images))]
    weight_sum_image = np.sum(weighted_images, axis=0)
    radiance_sum_image = np.sum(radiance_images, axis=0)

    idx_invalid = (weight_sum_image == 0)
    weight_sum_image[idx_invalid] = 1
    radiance_sum_image[idx_invalid] = 0

    return np.divide(radiance_sum_image/255., weight_sum_image), idx_invalid, weight_sum_image


def demosaic(im_raw, dtype=np.uint8):
    im_raw = im_raw.squeeze()
    N2, M2 = im_raw.shape

    N = N2 // 2
    M = M2 // 2
    im = np.zeros((N, M, 4, 3), dtype=dtype)

    for p in range(4):
        # 90, 45, 135 ,0 w.r.t. the horizontal axis
        py = np.uint8(p // 2)
        px = np.mod(p, 2)

        im_ = np.zeros((N, M), dtype=dtype)
        im_[0::2, 0::2] = im_raw[py::4, px::4]  # R
        im_[0::2, 1::2] = im_raw[py::4, 2 + px::4]  # G
        im_[1::2, 0::2] = im_raw[2 + py::4, px::4]  # G
        im_[1::2, 1::2] = im_raw[2 + py::4, 2 + px::4]  # B
        im[:, :, p, :] = cv2.demosaicing(im_, cv2.COLOR_BayerRG2BGR)

    return im


def polarRaw2DiffuseSpecular(im):
    ''' 
    im: N, M, 4, 3: 
    
    '''
    im0 = im[:,:,0,:]
    im90 = im[:,:,3,:]
    im45 = im[:,:,1,:]
    im135 = im[:,:,2,:]
    
    s0 = np.clip((im0 + im90 + im45 + im135)/2, 0, None)
    s1 = np.clip(im0 - im90, 0, None)
    s2 = np.clip(2*im45 - s0, 0, None)
    
    specular = np.sqrt(s1**2 + s2**2)
    diffuse = np.clip(s0 - specular, 0, None)
    return diffuse, specular


def decomposition(im):
    ''' 
    im: N, M, 4, 3: 
    
    '''
    im0 = im[:,:,0,:]
    im90 = im[:,:,3,:]
    im45 = im[:,:,1,:]
    im135 = im[:,:,2,:]
    
    s0 = np.clip((im0 + im90 + im45 + im135)/2, 0, None)
    s1 = np.clip(im0 - im90, 0, None)
    s2 = np.clip(2*im45 - s0, 0, None)
    
    diffuse = im90
    # specular = np.clip(s0-diffuse, 0, None)
    # specular = np.clip(s0-diffuse*2, 0, None)
    specular = np.clip(im0-diffuse, 0, None)
    
    return diffuse, specular


def process_HDR(path, exposure_list):
    if not os.path.exists(os.path.join(path, f'polar0')):
        os.mkdir(os.path.join(path, f'polar0'))
    if not os.path.exists(os.path.join(path, f'polar1')):
        os.mkdir(os.path.join(path, f'polar1'))
    if not os.path.exists(os.path.join(path, f'polar2')):
        os.mkdir(os.path.join(path, f'polar2'))
    if not os.path.exists(os.path.join(path, f'polar3')):
        os.mkdir(os.path.join(path, f'polar3'))
        
    if not os.path.exists(os.path.join(path, f'diffuse')):
        os.mkdir(os.path.join(path, f'diffuse'))
    if not os.path.exists(os.path.join(path, f'specular')):
        os.mkdir(os.path.join(path, f'specular'))
    if not os.path.exists(os.path.join(path, f'diffuseNspecular')):
        os.mkdir(os.path.join(path, f'diffuseNspecular'))

    # Load the calibration parameters from the file
    if path.endswith('main'):
        # calibration_path = 'main/calibration_data.npz'

        calibration_path = 'calibration_data.npz'
        
        black_level_noise_path = 'C:/Users/owner/Desktop/20240704_black/main'
    elif path.endswith('side'):
        calibration_path = 'side/calibration_data.npz'
        black_level_noise_path = 'C:/Users/owner/Desktop/20240704_black/side'
    calibration_data = np.load(calibration_path)
    intrinsic_matrix = calibration_data['K']
    distortion_coefficients = calibration_data['distCoeffs']
    
    
    # black_level_noise_path = 'C:/Users/owner/Desktop/20240628_black_0'
    # black_level_noise_path = 'C:/Users/owner/Desktop/20240704_black/'
    black_level_noise_path = path
    black_level_noise_list = []
    png_file_names_list = []
    for exposure in exposure_list:
        black_level_noise = Image.open(os.path.join(black_level_noise_path, f'black_{int(exposure)}.png'))
        black_level_noise = np.array(black_level_noise).astype(np.float64)/255.
        black_level_noise_list.append(black_level_noise)
        png_file_names = [f for f in os.listdir(path) if f.endswith(f'_{exposure}.png')]
        png_file_names.sort()
        print(png_file_names)
        png_file_names_list.append(png_file_names)
    print(png_file_names_list)
    print(exposure_list)
        

    scene_N = len(png_file_names)
    exposure_N = len(exposure_list)
    for i in range(scene_N):
        
        ldr_images = []     
        for j in range(exposure_N):
            
            im_ldr_raw = Image.open(os.path.join(path, png_file_names_list[j][i]))
            im_ldr_raw = np.array(im_ldr_raw).astype(np.float64)/255.
            temp = im_ldr_raw - black_level_noise_list[j] 
            # temp = im_ldr_raw
            
            temp = np.clip(temp, 0, None)     
            # temp = temp.astype(np.float64)/255      
            ldr_images.append(temp)
        
        ldr_polar0 = []
        ldr_polar1 = []
        ldr_polar2 = []
        ldr_polar3 = []
        ldr_diffuse = []
        ldr_specular = []
        ldr_diffuseNspecular = []
        for idx, ldr_image in enumerate(ldr_images):
            im_raw = (ldr_image*65535).astype(np.uint16) # ???
        
            #### demosaic with polarization ####
            im_raw = im_raw[:,:, np.newaxis]
            
            # break
            im = demosaic(im_raw, dtype=np.uint16)
            im = im.astype(np.float64)
            ####################################
            
            processed_im = np.zeros_like(im, dtype=np.float64)
            for polar in range(4):
                ############ undistortion ########## 나중으로 미루기!!!!!!!!!!!! <-- image format에서만 동작해서 hdr 이미지는 안됨.
                # Undistort the image using the calibration matrix and distortion coefficients
                undistorted_im = cv2.undistort(im[:,:,polar,:], intrinsic_matrix, distortion_coefficients)
                undistorted_im = undistorted_im/65535

                ############ white balance #########
                balanced_im = (undistorted_im).astype(np.float64) #* scailing_factor

                # Save
                processed_im[:,:,polar,:] = balanced_im
                
                
            diffuse, specular = polarRaw2DiffuseSpecular(processed_im)
            
            diffuseNspecular = np.sum(processed_im, axis=2)/4
            ldr_diffuse.append((diffuse*255).astype(np.uint8))
            ldr_specular.append((specular*255).astype(np.uint8))
            ldr_diffuseNspecular.append((diffuseNspecular*255).astype(np.uint8))
            
            ldr_polar0.append((processed_im[:,:,0]*255).astype(np.uint8))
            ldr_polar1.append((processed_im[:,:,1]*255).astype(np.uint8))
            ldr_polar2.append((processed_im[:,:,2]*255).astype(np.uint8))
            ldr_polar3.append((processed_im[:,:,3]*255).astype(np.uint8))


        hdr_diffuse, idx_invalid, weight_sum_image = make_hdr(ldr_diffuse, exposure_list)
        hdr_specular, idx_invalid, weight_sum_image = make_hdr(ldr_specular, exposure_list)
        hdr_diffuseNspecular, idx_invalid, weight_sum_image = make_hdr(ldr_diffuseNspecular, exposure_list)

        hdr_polar0, idx_invalid, weight_sum_image = make_hdr(ldr_polar0, exposure_list)
        hdr_polar1, idx_invalid, weight_sum_image = make_hdr(ldr_polar1, exposure_list)
        hdr_polar2, idx_invalid, weight_sum_image = make_hdr(ldr_polar2, exposure_list)
        hdr_polar3, idx_invalid, weight_sum_image = make_hdr(ldr_polar3, exposure_list)
        
        fn = png_file_names_list[j][i]
        last_ind = fn.rfind("_")
        fn_hdr = fn[:last_ind] +  ".npy"
        
        np.save(os.path.join(path, f'diffuse/{fn_hdr}'), (hdr_diffuse)[:,:,::-1])
        np.save(os.path.join(path, f'specular/{fn_hdr}'), (hdr_specular)[:,:,::-1])
        np.save(os.path.join(path, f'diffuseNspecular/{fn_hdr}'), (hdr_diffuseNspecular)[:,:,::-1])
        
        np.save(os.path.join(path, f'polar0/{fn_hdr}'), (hdr_polar0)[:,:,::-1])
        np.save(os.path.join(path, f'polar1/{fn_hdr}'), (hdr_polar1)[:,:,::-1])
        np.save(os.path.join(path, f'polar2/{fn_hdr}'), (hdr_polar2)[:,:,::-1])
        np.save(os.path.join(path, f'polar3/{fn_hdr}'), (hdr_polar3)[:,:,::-1])
        
            
        fn = png_file_names_list[j][i]
        last_ind = fn.rfind("_")
        fn_hdr = fn[:last_ind] +  ".png"
        
        # cv2.imwrite(os.path.join(path, f'diffuse/{fn_hdr}'), (hdr_diffuse/hdr_diffuse.max()*255).astype(np.uint8)[:,:,::-1])
        # cv2.imwrite(os.path.join(path, f'specular/{fn_hdr}'), (hdr_specular/hdr_specular.max()*255).astype(np.uint8)[:,:,::-1])
        # cv2.imwrite(os.path.join(path, f'diffuseNspecular/{fn_hdr}'), (hdr_diffuseNspecular/hdr_diffuseNspecular.max()*255).astype(np.uint8)[:,:,::-1])
        
        # cv2.imwrite(os.path.join(path, f'polar0/{fn_hdr}'), (hdr_polar0/hdr_polar0.max()*255).astype(np.uint8)[:,:,::-1])
        # cv2.imwrite(os.path.join(path, f'polar1/{fn_hdr}'), (hdr_polar1/hdr_polar1.max()*255).astype(np.uint8)[:,:,::-1])
        # cv2.imwrite(os.path.join(path, f'polar2/{fn_hdr}'), (hdr_polar2/hdr_polar2.max()*255).astype(np.uint8)[:,:,::-1])
        # cv2.imwrite(os.path.join(path, f'polar3/{fn_hdr}'), (hdr_polar3/hdr_polar3.max()*255).astype(np.uint8)[:,:,::-1])
        
            

        cv2.imwrite(os.path.join(path, f'diffuse/{fn_hdr}'), (np.clip(hdr_diffuse, 0., 1.)*255).astype(np.uint8)[:,:,::-1])
        cv2.imwrite(os.path.join(path, f'specular/{fn_hdr}'), (np.clip(hdr_specular, 0., 1.)*255).astype(np.uint8)[:,:,::-1])
        cv2.imwrite(os.path.join(path, f'diffuseNspecular/{fn_hdr}'), (np.clip(hdr_diffuseNspecular, 0., 1.)*255).astype(np.uint8)[:,:,::-1])
        
        cv2.imwrite(os.path.join(path, f'polar0/{fn_hdr}'), (np.clip(hdr_polar0, 0., 1.)*255).astype(np.uint8)[:,:,::-1])
        cv2.imwrite(os.path.join(path, f'polar1/{fn_hdr}'), (np.clip(hdr_polar1, 0., 1.)*255).astype(np.uint8)[:,:,::-1])
        cv2.imwrite(os.path.join(path, f'polar2/{fn_hdr}'), (np.clip(hdr_polar2, 0., 1.)*255).astype(np.uint8)[:,:,::-1])
        cv2.imwrite(os.path.join(path, f'polar3/{fn_hdr}'), (np.clip(hdr_polar3, 0., 1.)*255).astype(np.uint8)[:,:,::-1])
        
            

        
        # cv2.imwrite(os.path.join(path, f'diffuse/{fn_hdr}'), (hdr_diffuse/0.25*255).astype(np.uint8)[:,:,::-1])
        # cv2.imwrite(os.path.join(path, f'specular/{fn_hdr}'), (hdr_specular/0.25*255).astype(np.uint8)[:,:,::-1])
        # cv2.imwrite(os.path.join(path, f'diffuseNspecular/{fn_hdr}'), (hdr_diffuseNspecular/0.25*255).astype(np.uint8)[:,:,::-1])
        
        # cv2.imwrite(os.path.join(path, f'polar0/{fn_hdr}'), (hdr_polar0/0.25*255).astype(np.uint8)[:,:,::-1])
        # cv2.imwrite(os.path.join(path, f'polar1/{fn_hdr}'), (hdr_polar1/0.25*255).astype(np.uint8)[:,:,::-1])
        # cv2.imwrite(os.path.join(path, f'polar2/{fn_hdr}'), (hdr_polar2/0.25*255).astype(np.uint8)[:,:,::-1])
        # cv2.imwrite(os.path.join(path, f'polar3/{fn_hdr}'), (hdr_polar3/0.25*255).astype(np.uint8)[:,:,::-1])
        
      
def process_LDRs(path, exposure_list):
    
    for exposure in exposure_list:
        process_LDR(path, exposure)     


def process_LDR(path, exposure):
    
    dir_name = str(exposure/1000)+'ms'  
    if not os.path.exists(os.path.join(path, dir_name)):
        os.mkdir(os.path.join(path, dir_name))
    
    for i in range(4):
        if not os.path.exists(os.path.join(path, f'{dir_name}/polar{i}')):
            os.mkdir(os.path.join(path, f'{dir_name}/polar{i}'))
    if not os.path.exists(os.path.join(path, f'{dir_name}/diffuse')):
        os.mkdir(os.path.join(path, f'{dir_name}/diffuse'))
    if not os.path.exists(os.path.join(path, f'{dir_name}/specular')):
        os.mkdir(os.path.join(path, f'{dir_name}/specular'))
    if not os.path.exists(os.path.join(path, f'{dir_name}/diffuseNspecular')):
        os.mkdir(os.path.join(path, f'{dir_name}/diffuseNspecular'))
        
        
    # Load the calibration parameters from the file
    if path.endswith('main'):
        calibration_path = './CameraParameters_main.mat'
        black_level_noise_path = 'C:/Users/owner/Desktop/20240704_black/main'
    elif path.endswith('side'):
        calibration_path = './CameraParameters_side.mat'
        black_level_noise_path = 'C:/Users/owner/Desktop/20240704_black/side'
        
    CameraParameters = scipy.io.loadmat(calibration_path)
    K, D = (CameraParameters['k'], CameraParameters['dist'].T)
    
    # black_level_noise_path = path
    black_level_noise = Image.open(os.path.join(black_level_noise_path, f'black_{int(exposure)}.png'))
    black_level_noise = np.array(black_level_noise)

    png_file_names = [f for f in os.listdir(path) if f.endswith(f'_{exposure}.png')]
    png_file_names.sort()
    
    for data in png_file_names:
        file_path = os.path.join(path, data)

        im_raw = Image.open(file_path)
        im_raw = np.array(im_raw)
        temp = im_raw - black_level_noise
        temp = np.clip(temp, 0, None)    
         
        #### demosaic with polarization ####
        im_raw = im_raw[:,:, np.newaxis]
        
        # break
        im = demosaic(im_raw)
        # im = demosaic(im_raw.astype(np.uint16)*255, dtype=np.uint16)
        ####################################
        
        processed_im = np.zeros_like(im, dtype=np.float64)
        for polar in range(4):
            ############ undistortion ##########
            # Undistort the image using the calibration matrix and distortion coefficients
            undistorted_im = cv2.undistort(im[:,:,polar,:], K, D)
            undistorted_im = undistorted_im/255.
            # undistorted_im = im[:,:,polar,:]/255
            balanced_im = (undistorted_im).astype(np.float64) #* scailing_factor          

            # Save
            processed_im[:,:,polar,:] = balanced_im
            
        diffuse, specular = decomposition(processed_im)
        # diffuse, specular = polarRaw2DiffuseSpecular(processed_im)
        
        last_ind = data.rfind("_")
        data = data[:last_ind] +  ".png"
        
        im_save = Image.fromarray((np.clip(diffuse, 0, 1.)*255).astype(np.uint8))
        im_save.save(os.path.join(path, f'{dir_name}/diffuse/{data}'))
        im_save = Image.fromarray((np.clip(specular, 0, 1.)*255).astype(np.uint8))
        im_save.save(os.path.join(path, f'{dir_name}/specular/{data}'))
        
        im_save = Image.fromarray((np.clip(np.sum(processed_im, axis=2)/4, 0, 1.)*255).astype(np.uint8))
        im_save.save(os.path.join(path, f'{dir_name}/diffuseNspecular/{data}'))
            
        for i in range(4):
            im_save = Image.fromarray((np.clip(processed_im[:,:,i,:], 0,1.)*255).astype(np.uint8))
            im_save.save(os.path.join(path, f'{dir_name}/polar{i}/{data}'))


def white_balance(path, colorchecker):
    return