import torch
import torch.nn.functional as F


class LightDeskRenderer:
    def __init__(self, opt):
        self.opt = opt

    def render(self, basis_images, monitor_light_pattern_nonlinear, camera_gain):
        B, RC, H, W, ch = basis_images.shape
        # basis_image: (B,R*C,H,W,3)
        monitor_light_radiance = torch.sigmoid(monitor_light_pattern_nonlinear) ** self.opt.monitor_gamma
        # monitor_light_radiance = monitor_light_pattern_nonlinear ** self.opt.monitor_gamma
        # monitor_light_radiance = torch.sigmoid(monitor_light_pattern_nonlinear) ** (1/2.06) # ** self.opt.monitor_gamma
        # monitor_light_radiance = torch.sigmoid(monitor_light_pattern_nonlinear) #** (1/2.06)# ** self.opt.monitor_gamma
        monitor_light_radiance = torch.flip(monitor_light_radiance, dims=[2])
        # print(monitor_light_radiance.shape)
        monitor_light_radiance = monitor_light_radiance.reshape(1, self.opt.light_N, RC, 3, 1)

        basis_images = basis_images.permute(0,1,4,2,3)
        # weights: [batch, lightN, R*C, 3, H*W]
        basis_images = basis_images.reshape(B, RC, ch, -1).unsqueeze(1)
        # print(basis_images.shape, monitor_light_radiance.shape)
        result = monitor_light_radiance * basis_images
        result = result.sum(axis=2)
        result = result.reshape(B, self.opt.light_N, 3, H, W)
        result = result.permute(0,1,3,4,2)


        gain_scalar = self.opt.cam_gain_base ** (torch.sigmoid(camera_gain)*48)
        
        # I_diffuse = gain_scalar * (self.opt.rendering_scalar * self.opt.cam_shutter_time * result)
        I_diffuse = gain_scalar * (self.opt.pattern_rendering_scalar * result)
        I_diffuse = torch.clamp(I_diffuse, 1e-8, 1)

        return I_diffuse
  

    def render_Cook_Torrance(self, n, l, v, da, sa, rg, camera_gain, coeff, falloff_scalar, backlight):
        # patterns:(N,R,C,3)
        # n:(B,3,H,W), l:(R,C,B,H,W,3), v:(B,3,H,W)
        # da:(B,basis,3,H,W), sa:(B,basis,1,H,W), rg:(B,basis,1,H,W)
        # coeff:(B,basis,W), falloff_scalar:(R,C,B,H,W)
        
        # RAW rendering
        gain_scalar = self.opt.cam_gain_base ** (torch.sigmoid(camera_gain)*48)
        rendering_scalar = gain_scalar * self.opt.GGX_rendering_scalar 
        
        r, c, B, h, w, _ = l.shape
        ch = 3 # color channel
        num_basis = sa.shape[1]
        ch_s = sa.shape[2]
        
        n = n.view(B, 3, 1, h*w)
        l = torch.flip(l, dims=[1])
        l = l.permute(2,5,0,1,3,4)
        l = l.reshape(B, 3, r*c, -1)
        v = v.view(B, 3, 1, -1)

        da = da.view(B, num_basis, ch, 1, -1)
        sa = sa.view(B, num_basis, ch_s, 1, -1)
        rg = rg.view(B, num_basis, 1, 1, -1)
        coeff = coeff.view(B, num_basis, 1, 1, -1)

        falloff_scalar = torch.flip(falloff_scalar, dims=[1])
        falloff_scalar = falloff_scalar.permute(2,0,1,3,4)
        falloff_scalar = falloff_scalar.reshape(B, 1, r*c, h*w)
        
        # nl:(B,R*C,H*W)
        # fd: (B, #basis, ch, 1, 1)        -coeff-> (B, ch, 1, H*W)      -nl-> (B, ch, R*C, H*W)
        # fs: (B, #basis, ch(1), r*c, H*W) -coeff-> (B, ch(1), r*c, H*W) -nl-> (B, ch(1), r*c, H*W)
        # nh: (B, 1,        (1), r*c, H*W)
        nl = torch.clamp(torch.sum(n * l, 1), 0, 1)
        fd = da
        fs, nh = self.compute_specular(l.unsqueeze(1), v.unsqueeze(1), n.unsqueeze(1), nl.unsqueeze(1).unsqueeze(1), sa, rg)
        fd = torch.sum(fd*coeff, dim=1)
        fs = torch.sum(fs*coeff, dim=1)
        fd = (nl*fd)*rendering_scalar*falloff_scalar
        fs = (nl*fs)*rendering_scalar*falloff_scalar
        
        # fd: (B, ch, R*C, H, W)
        # fs: (B, 1 , R*C, H, W)
        fd = fd.view(B, ch, r*c, h, w)
        fs = fs.view(B, ch_s, r*c, h, w)
        
        backlight_pattern = (backlight) ** self.opt.monitor_gamma
        backlight_pattern = backlight_pattern.repeat(1,1,3).unsqueeze(0)
        backlight_pattern = torch.flip(backlight_pattern, dims=[2])
        backlight_pattern = backlight_pattern.permute(0,3,1,2)
        backlight_pattern = backlight_pattern.reshape(1, 1, ch, r*c, 1, 1)
        backlight_effect = torch.sum((fd+fs).unsqueeze(1)*backlight_pattern, dim=3) # (B, 1, ch, h, w)
        
        OLAT = (fd+fs).permute(0,2,3,4,1) # (B, r*c, h, w, ch)
        OLAT = torch.clamp(OLAT*0.9, 0, None)
        OLAT = backlight_effect.permute(0,1,3,4,2) + OLAT
        
        OLAT_fd = fd.permute(0,2,3,4,1)
        OLAT_fs = fs.permute(0,2,3,4,1)
        
        OLAT = torch.clamp(OLAT, 0, 1)
        
        return OLAT_fd, OLAT_fs, OLAT, nl, nh # (B, N, h, w, ch)


    def compute_specular(self, i, o, n, ni, rho_s, m, Fresnel=0.98):
        # i:(B, 1, 3, R*C, H*W)
        # o:(B, 1, 3,  1,  H*W)
        # n:(B, 1, 3,  1,  H*W)
        # sa, rg: (B, num_basis, 1, 1, H*W)
        
        def dot(x1, x2):
            return torch.sum(x1 * x2, dim=2, keepdim=True)

        def compute_Beckmann_NDF(m, cos_th_h, tan_th_h):
            """
            Compute the Beckmann's normal distribution function

            Args
                m: roughness
                    dim: [# of px, 1]
                th_h: half-wave angle in rad
                    dim: [# of px, 1]
            Return
                D: Beckmann's distribution
                    dim: [# of px, 1]
            """

            tan_th_h[tan_th_h > 1e3] = 1e3

            D = torch.exp(-((-tan_th_h / m) ** 2)) / ((m ** 2) * (cos_th_h ** 4) + 1e-8)  #
            return D

        def compute_GGX_NDF(m, nh):
            D = m**2 / torch.clamp((4*((nh**2)*(m**2-1)+1)**2), 1e-8, None)
            
            return D
        
        def compute_Schlick_GGX_SMF(m, ni, no):
            k = m/2
            
            G1 = ni/(ni*(1-k)+k).clamp(min=1e-8)
            G2 = no/(no*(1-k)+k).clamp(min=1e-8)
            
            return G1*G2

        def compute_Smith_SMF(m, tan_th_i, tan_th_o):
            """
            Compute the Smith's shading/masking function

            Args
                m: roughness
                    dim: [# of px, 1]
                tan_th_i: incident angle in rad
                    dim: [# of px, 1]
                tan_th_o: exitant angle in rad
                    dim: [# of px, 1]
            Return
                S: Smith's distribution
                    dim: [# of px, 1]
            """
            G = (2 / (1 + torch.sqrt(1 + (m ** 2) * (tan_th_i ** 2)))) * (
                        2 / (1 + torch.sqrt(1 + (m ** 2) * (tan_th_o ** 2))))
            return G
        
        # compute useful angles: (B, 1, 1, R*C, H*W)
        h = i + o
        h = F.normalize(h, dim=2)
        no, nh = dot(o, n), dot(h, n)
        no, nh = torch.clamp(no, 0,1), torch.clamp(nh, 0,1)

        # -------------- specular ---------------
        # micro-facet constants
        d = compute_GGX_NDF(m, nh)
        g = compute_Schlick_GGX_SMF(m, ni, no)
        fs = rho_s * Fresnel * d * g / torch.clamp(4 * no * ni, 1e-8, None)
        
        return fs, nh


    def render_Cook_Torrance_sphere(self, n, l, v, da, sa, rg):
        # [h, w] <=> [pixels, 1]
        
        # n [3, h, w]
        # l [N, 3, h, w]
        # v [3, h, w]
        # da [3]
        # sa [1]
        # rg [1]
        
        _, N, h, w = l.shape 
        
        n = F.normalize(n, p=2, dim=0)
        n = n.reshape(3, 1, h*w)
        
        l = l.reshape(3, N, -1)
        v = v.reshape(3, 1, -1)

        da = da.reshape(3, 1, -1)
        sa = sa.reshape(1, 1, -1)
        rg = rg.reshape(1, 1, -1)

        nl = torch.clamp(torch.sum(n * l, 0), 0, None)
        fd = da
        fs = self.compute_specular_sphere(l, v, n, sa, rg)
        
        fd = torch.sum(nl*fd, dim=1)
        fs = torch.sum(nl*fs, dim=1)
        
        rendered = (fd + fs)
        rendered = torch.clamp(rendered, 1e-8, 1)
        
        return nl, fd, fs, rendered # (ch, h, w)


    def compute_specular_sphere(self, i, o, n, rho_s, m, Fresnel=0.98, F0=1.5):
        """
        Compute the diffuse/specular Cook-Torrance BRDF model for three channels

        Args
            i: incident vector
                dim: [# of px, 3]
            o: exitant vector
                dim: [# of px, 3]
            n: surface normal
                dim: [# of px, 3]
            rho_s: specular albedo
                dim: [# of px, 3]
            m: roughness
                dim: [# of px, 1]
            F: Fresnel term
                dim: [# of px, 3]

        Return
            f: BRDF
                dim: [# of px, 3]
            fd: diffuse BRDF
                dim: [# of px, 3]
            fs: specular BRDF
                dim: [# of px, 3]
        """
        
        def dot(x1, x2):
            return torch.sum(x1 * x2, dim=0, keepdim=True)

        def compute_Beckmann_NDF(m, cos_th_h, tan_th_h):
            """
            Compute the Beckmann's normal distribution function

            Args
                m: roughness
                    dim: [# of px, 1]
                th_h: half-wave angle in rad
                    dim: [# of px, 1]
            Return
                D: Beckmann's distribution
                    dim: [# of px, 1]
            """

            tan_th_h[tan_th_h > 1e3] = 1e3

            D = torch.exp(-((-tan_th_h / m) ** 2)) / ((m ** 2) * (cos_th_h ** 4) + 1e-8)  #
            return D

        def compute_GGX_NDF(m, cos_th_h, tan_th_h):

            tan_th_h[tan_th_h > 1e3] = 1e3

            D = m**2 / (4 * cos_th_h**4 * (m**2 + tan_th_h**2)**2 + 1e-8)

            return D

        def compute_Smith_SMF(m, tan_th_i, tan_th_o):
            """
            Compute the Smith's shading/masking function

            Args
                m: roughness
                    dim: [# of px, 1]
                tan_th_i: incident angle in rad
                    dim: [# of px, 1]
                tan_th_o: exitant angle in rad
                    dim: [# of px, 1]
            Return
                S: Smith's distribution
                    dim: [# of px, 1]
            """
            G = (2 / (1 + torch.sqrt(1 + (m ** 2) * (tan_th_i ** 2)))) * (
                        2 / (1 + torch.sqrt(1 + (m ** 2) * (tan_th_o ** 2))))
            return G

        def compute_Schlick_GGX_SMF(m, ni, no):
            k = m/2
            
            G1 = ni/(ni*(1-k)+k).clamp(min=1e-8)
            G2 = no/(no*(1-k)+k).clamp(min=1e-8)
            
            return G1*G2
        
        def compute_Schlick_Fresnel(F0, no):
            F = F0 + (1-F0)*(1-no)**5
            return F

        # compute useful angles
        h = i + o
        h = F.normalize(h, dim=0)
        ni, no, nh = dot(i, n), dot(o, n), dot(h, n)
        ni, no, nh = torch.clamp(ni, 1e-8, 1 - 1e-8), torch.clamp(no, 1e-8, 1 - 1e-8), torch.clamp(nh, 1e-8, 1 - 1e-8)


        # -------------- specular ---------------
        # micro-facet constants
        tan_th_h = torch.sqrt(torch.abs(1 - nh ** 2)) / (nh + 1e-8)
        d = compute_GGX_NDF(m, nh, tan_th_h)
        # tan_th_i = torch.sqrt(1 - ni ** 2 + 1e-8) / (ni + 1e-8)
        # tan_th_o = torch.sqrt(1 - no ** 2 + 1e-8) / (no + 1e-8)
        g = compute_Schlick_GGX_SMF(m, ni, no)
        # g = compute_Smith_SMF(m, tan_th_i, tan_th_o)
        fs = rho_s * Fresnel * d * g / (4 * no * ni + 1e-8)

        return fs

