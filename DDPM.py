import torch
import torch.nn as nn
import torch.nn.functional as F
from ResUNet import ConditionalUnet
#from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConditionalDDPM(nn.Module):
    def __init__(self, dmconfig):
        super().__init__()
        self.dmconfig = dmconfig
        self.loss_fn = nn.MSELoss()
        self.network = ConditionalUnet(1, self.dmconfig.num_feat, self.dmconfig.num_classes)

    def scheduler(self, t_s):
        beta_1, beta_T, T = self.dmconfig.beta_1, self.dmconfig.beta_T, self.dmconfig.T
        # ==================================================== #
        # YOUR CODE HERE:
        #   Inputs:
        #       t_s: the input time steps, with shape (B,1). 
        #   Outputs:
        #       one dictionary containing the variance schedule
        #       $\beta_t$ along with other potentially useful constants.       
        device = t_s.device

        
        beta_all = torch.linspace(beta_1, beta_T, T, device=device)  # float32 by default
        alpha_all = 1.0 - beta_all
        alpha_bar_all = torch.cumprod(alpha_all, dim=0)

        
        idx = (t_s.view(-1).long() - 1)  # shape (B,)

        
        beta_t   = beta_all[idx]
        sqrt_beta_t = beta_t.sqrt()

        alpha_t  = alpha_all[idx]
        oneover_sqrt_alpha = 1.0 / alpha_t.sqrt()

        alpha_t_bar = alpha_bar_all[idx]
        sqrt_alpha_bar = alpha_t_bar.sqrt()
        sqrt_oneminus_alpha_bar = (1.0 - alpha_t_bar).sqrt()

        


        # ==================================================== #
        return {
            'beta_t': beta_t,
            'sqrt_beta_t': sqrt_beta_t,
            'alpha_t': alpha_t,
            'sqrt_alpha_bar': sqrt_alpha_bar,
            'oneover_sqrt_alpha': oneover_sqrt_alpha,
            'alpha_t_bar': alpha_t_bar,
            'sqrt_oneminus_alpha_bar': sqrt_oneminus_alpha_bar
        }

    def forward(self, images, conditions):
        T = self.dmconfig.T
        noise_loss = None
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given training algorithm.
        #   Inputs:
        #       images: real images from the dataset, with size (B,1,28,28).
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #   Outputs:
        #       noise_loss: loss computed by the self.loss_fn function  .  

        B = images.size(0)
        device = images.device

        
        cond = F.one_hot(conditions.long(),
                         num_classes=self.dmconfig.num_classes
                        ).float().to(device)             # (B,10)
        
        mask = torch.rand(B, device=device) < self.dmconfig.mask_p
        cond[mask] = self.dmconfig.condition_mask_value

        
        t = torch.randint(1, T + 1, (B, 1), device=device)  # (B,1)
        sched = self.scheduler(t)                          # dict of schedules

        
        sqrt_ab = sched['sqrt_alpha_bar'].view(B, 1, 1, 1)
        sqrt_omb = sched['sqrt_oneminus_alpha_bar'].view(B, 1, 1, 1)
        noise   = torch.randn_like(images)
        x_t     = sqrt_ab * images + sqrt_omb * noise

        
        t_norm = (t.float() / T).view(B, 1, 1, 1)

        
        pred_noise = self.network(x_t, t_norm, cond)

        
        noise_loss = self.loss_fn(pred_noise, noise)



        # ==================================================== #
        
        return noise_loss
    
    
    def sample(self, conditions, omega):
        T = self.dmconfig.T
        X_t = None
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given sampling algorithm.
        #   Inputs:
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #       omega: conditional guidance weight.
        #   Outputs:
        #       generated_images  

        T      = self.dmconfig.T
        device = next(self.network.parameters()).device
        B      = conditions.size(0)

        
        cond    = F.one_hot(conditions.long(), self.dmconfig.num_classes) \
                      .float().to(device)  # (B, num_classes)
        cond_un = torch.full_like(cond, self.dmconfig.condition_mask_value)

        
        x_t = torch.randn(
            B,
            self.network.in_channels,
            *self.dmconfig.input_dim,
            device=device
        )

        
        self.network.eval()
        with torch.no_grad():
            for t in range(T, 0, -1):
                # build a batch of identical timesteps
                t_b = torch.full((B,1), t, device=device)

                # fetch schedule
                sched    = self.scheduler(t_b)
                beta_t   = sched['beta_t'].view(B,1,1,1)
                sqrt_bt  = sched['sqrt_beta_t'].view(B,1,1,1)
                inv_sqa  = sched['oneover_sqrt_alpha'].view(B,1,1,1)
                sqrt_omb = sched['sqrt_oneminus_alpha_bar'].view(B,1,1,1)
                t_norm   = (t_b.float() / T).view(B,1,1,1)

                # εθ(x_t, c) and εθ(x_t) → guidance
                eps_c = self.network(x_t, t_norm, cond)
                eps_u = self.network(x_t, t_norm, cond_un)
                eps   = (1.0 + omega)*eps_c - omega*eps_u

                # compute x_{t-1}
                if t > 1:
                    z = torch.randn_like(x_t)
                else:
                    z = torch.zeros_like(x_t)
                x_t = inv_sqa * (x_t - (beta_t / sqrt_omb)*eps) + sqrt_bt * z



        # ==================================================== #
        generated_images = (x_t * 0.3081 + 0.1307).clamp(0,1) # denormalize the output images
        return generated_images