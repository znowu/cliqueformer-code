import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from architectures.backbones import MLP

import data.extras as extras


class DDOM(nn.Module):

    def __init__(self, input_dim, hidden_dims=2*(1024,), gamma=2., n_timesteps=1000, 
                 act=nn.ReLU(), K_factor=0.1, N_bins=64, temp=1e-1, uncond_rate=0.15, lr=1e-3):

        super().__init__()

        self.input_dim = input_dim 
        self.gamma = gamma 
        self.act = act 
        self.temp = temp 
        self.n_timesteps = n_timesteps
        self.uncond_rate = uncond_rate
        self.lr = lr

        self.model = MLP(input_dim + 3, input_dim, hidden_dims, act)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        self.timesteps = torch.arange(n_timesteps + 1)

        log_beta = -1e-2 / n_timesteps - 0.5 * (2 - 1e-2) * (2 * self.timesteps - 1) / n_timesteps**2 
        self.beta = 1 - torch.exp(log_beta) 
        self.beta[0] = 0.
        self.alpha = 1 - self.beta 
        self.alpha_bar = torch.cumprod(self.alpha, -1)

        self.K_factor = K_factor
        self.N_bins = N_bins
        self.temp = temp
        self.bins = None

    def get_bins(self, y):
        self.bins =extras.compute_bin_weights(y, self.K_factor, self.N_bins, self.temp)

    def forward(self, x, y, m, t):
        #
        # Reshape the label, mask, and time inputs
        #
        y = y.reshape(-1, 1)
        m = m.reshape(-1, 1)
        t = t.reshape(-1, 1) / self.n_timesteps

        #
        # Mask out the specified labels
        #
        y = y * m

        #
        # Concatenate and run the model
        #
        x = torch.cat([x, y, m, t], dim=-1)

        return self.model(x)

    def denoising_loss(self, x, y, weights):
        #
        # Read off shapes of the input
        #
        B = x.shape[0]
        shape = x.shape[1:]

        #
        # Sample the time and the noise variables
        #
        t = torch.randint(1, len(self.timesteps), (B,))
        m = 1. * (torch.rand(B) > self.uncond_rate).to(x.device)
        noise = torch.randn(x.shape).to(x.device)

        #
        # Compute the noisy input
        #
        alpha_bar = self.alpha_bar[t].reshape((B,) + len(shape) * (1,))
        alpha_bar = alpha_bar.to(x.device)
        noisy_x = torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * noise 

        #
        # Calculate the denoising loss 
        #
        pred_noise = self.forward(noisy_x, y, m, t.to(x.device))
        weights = weights / weights.mean()
        error = ((pred_noise - noise)**2).sum(-1)
        loss = (weights * error).mean()

        return loss 
    
    def training_step(self, x, y):
        
        bins = extras.move_to_device(self.bins, x.device)
        weights = extras.assign_bin_weights(y, bins)

        loss = self.denoising_loss(x, y, weights)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {
            'loss': loss.item(),
            'loss_pd': loss.item() / self.input_dim
        }
    
    def eval_step(self, x, y):

        self.eval()

        with torch.no_grad():
            bins = extras.move_to_device(self.bins, x.device)
            weights = extras.assign_bin_weights(y, bins)
            loss = self.denoising_loss(x, y, weights)

        return {
            'loss': loss.item(),
            'loss_pd': loss.item() / self.input_dim
        }
    
    def heun_step(self, x_t, y, t, t_prev):
        #
        # Convert timesteps to tensor format
        #
        t_tensor = torch.ones(x_t.shape[0], device=x_t.device) * t
        t_prev_tensor = torch.ones(x_t.shape[0], device=x_t.device) * t_prev
        
        #
        # Get coefficients
        #
        alpha_bar_t = self.alpha_bar[t]
        alpha_bar_prev = self.alpha_bar[t_prev]
        
        #
        # Create conditional and unconditional masks
        #
        cond_mask = torch.ones_like(y)
        uncond_mask = torch.zeros_like(y)
        
        #
        # Get conditional and unconditional predictions
        #
        eps_t_cond = self(x_t, y, cond_mask, t_tensor)
        eps_t_uncond = self(x_t, y, uncond_mask, t_tensor)
        
        #
        # Apply classifier-free guidance
        #
        eps_t = eps_t_cond + self.gamma * (eps_t_cond - eps_t_uncond)
        
        #
        # If this is the first step (t = n_timesteps), use Euler step only
        #
        if t == self.n_timesteps:
            x_0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_t) / torch.sqrt(alpha_bar_t)
            x_prev = torch.sqrt(alpha_bar_prev) * x_0_pred + torch.sqrt(1 - alpha_bar_prev) * eps_t
        
        else:
            #
            # Compute Euler step (predictor)
            #
            x_0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_t) / torch.sqrt(alpha_bar_t)
            x_prev_euler = torch.sqrt(alpha_bar_prev) * x_0_pred + torch.sqrt(1 - alpha_bar_prev) * eps_t
            
            #
            # Second prediction using the Euler prediction (corrector)
            #
            eps_prev_cond = self(x_prev_euler, y, cond_mask, t_prev_tensor)
            eps_prev_uncond = self(x_prev_euler, y, uncond_mask, t_prev_tensor)
            eps_prev = eps_prev_cond + self.gamma * (eps_prev_cond - eps_prev_uncond)
            
            #
            # Combine both predictions (Heun's correction step)
            #
            eps_combined = (eps_t + eps_prev) / 2
            x_0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_combined) / torch.sqrt(alpha_bar_t)
            x_prev = torch.sqrt(alpha_bar_prev) * x_0_pred + torch.sqrt(1 - alpha_bar_prev) * eps_combined
        
        #
        # Add noise 
        #
        sigma = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)).to(x_prev.device)
        noise = torch.randn_like(x_t).to(sigma.device)
        x_prev = x_prev + sigma * noise
    
        return x_prev

    def sample(self, batch_size, y, device="cuda"):

        x = torch.randn(batch_size, self.input_dim, device=device)
        y = y * torch.ones(batch_size).to(device)
        
        for t in range(self.n_timesteps, 1, -1):
            x = self.heun_step(x, y, t, t-1)
        
        return x
    

class DDOMDiscrete(DDOM):

    def __init__(self, n_input, input_dim, hidden_dims=2 * (1024, ), gamma=2, n_timesteps=1000, act=nn.ReLU(), 
                 K_factor=0.1, N_bins=64, temp=0.1, uncond_rate=0.15, lr=0.001):
        
        super().__init__(n_input * input_dim, hidden_dims, gamma, n_timesteps, act, K_factor, N_bins, temp, uncond_rate, lr)

        self.n_input = n_input
        self.input_dim = input_dim

    def forward(self, x, y, m, t):

        x = x.reshape(x.shape[0], -1)        
        x = super().forward(x, y, m, t)
        return x.reshape(x.shape[0], self.n_input, self.input_dim)
    
    def denoising_loss(self, x, y, weights):
        #
        # Read off shapes of the input
        #
        B = x.shape[0]
        shape = x.shape[1:]

        #
        # Sample the time and the noise variables
        #
        t = torch.randint(1, len(self.timesteps), (B,))
        m = 1. * (torch.rand(B) > self.uncond_rate).to(x.device)
        noise = torch.randn(x.shape).to(x.device)

        #
        # Compute the noisy input
        #
        alpha_bar = self.alpha_bar[t].reshape((B,) + len(shape) * (1,))
        alpha_bar = alpha_bar.to(x.device)
        noisy_x = torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * noise 

        #
        # Calculate the denoising loss 
        #
        pred_noise = self.forward(noisy_x, y, m, t.to(x.device))
        weights = weights / weights.mean()
        error = ((pred_noise - noise)**2).sum(-1).sum(-1)
        loss = (weights * error).mean()

        return loss 

    def sample(self, batch_size, y, device="cuda"):

        x = torch.randn(batch_size, self.n_input, self.input_dim, device=device)
        y = y * torch.ones(batch_size).to(device)

        for t in range(self.n_timesteps, 1, -1):
            x = self.heun_step(x, y, t, t-1)
        
        x = torch.argmax(x, dim=-1)
        x = F.one_hot(x, self.input_dim)
    
        return x
    