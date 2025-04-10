import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as Dist
from torch.autograd.functional import jacobian
from typing import Union
class GroupLinearLayer(nn.Module):
    """GroupLinearLayer computes N dinstinct linear transformations at once"""
    def __init__(
        self, 
        din: int, 
        dout: int, 
        num_blocks: int,
        diagonal: bool = False,
        hidden: Union[None, int] = None) -> None:
        """Group Linear Layer module

        Args:
            din: The feature dimension of input data.
            dout: The projected dimensions of data.
            num_blocks: The number of linear transformation to compute at once.
            diagonal: Whether transition matrix is diagonal
        """
        super(GroupLinearLayer, self).__init__()
        assert (hidden is None) or (type(hidden) == int)
        self.hidden = hidden
        self.diagonal = diagonal
        # Sparse transition already implements low-rank
        assert (bool(self.hidden) and self.diagonal) == False
        if diagonal:
            self.d = nn.Parameter(0.01 * torch.randn(num_blocks, dout))
        else:
            if hidden is None:
                self.w = nn.Parameter(0.01 * torch.randn(num_blocks, din, dout))
            else:
                assert isinstance(hidden, int)
                self.wh = nn.Parameter(0.01 * torch.randn(num_blocks, din, hidden))
                self.hw = nn.Parameter(0.01 * torch.randn(num_blocks, hidden, dout))

    def forward(
        self,
        x: torch.Tensor) -> torch.Tensor:
        if self.diagonal:
            w = torch.diag_embed(self.d)
            # x: [BS,num_blocks,din]->[num_blocks,BS,din]
            x = x.permute(1,0,2)
            x = torch.bmm(x, w)
            # x: [BS,num_blocks,dout]
            x = x.permute(1,0,2)
        elif self.hidden is None:
            x = x.permute(1,0,2)
            x = torch.bmm(x, self.w)
            # x: [BS,num_blocks,dout]
            x = x.permute(1,0,2)
        else:
            x = x.permute(1,0,2)
            # x: [num_blocks,BS,din]->[num_blocks,BS,hidden]
            x = torch.bmm(x, self.wh)           
            x = torch.bmm(x, self.hw)  
            x = x.permute(1,0,2)
        return x
    
    def get_weight_matrix(self):
        if self.diagonal:
            return torch.diag_embed(self.d)
        elif self.hidden is None:
            return self.w 
        else:
            return torch.matmul(self.wh, self.hw)
        
class NLayerLeakyMLP(nn.Module):

    def __init__(self, input_dim, output_dim, num_layers, hidden_dim, bias=True):
        super().__init__()
        layers = [ ]
        for l in range(num_layers):
            if l == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
                # layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                # layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: B, L, D
        return self.net(x) # B, L, D
    
class build_dist(nn.Module):

    def __init__(self, lag=1, z_dim=1024, num_layers=4, hidden_dim=512):
        super().__init__()
        self.z_dim = z_dim
        self.lag = lag
        self.f1 = nn.Linear(lag*z_dim, z_dim*2) # [512, 1024]
        self.f2 = nn.Linear(2*hidden_dim, hidden_dim) # [1024, 1024]

        self.net = NLayerLeakyMLP(input_dim=hidden_dim, 
                                  output_dim=z_dim*2, 
                                  num_layers=num_layers, 
                                  hidden_dim=hidden_dim)

    def forward(self, x):
        # [B, 2048] -> [B, 2048]
        zs = x[:,:self.lag*self.z_dim]
        distributions = self.f1(zs)
        enc = self.f2(x[:,self.lag*self.z_dim:])
        distributions = distributions + self.net(enc)
        return distributions
    
class MBDTransitionPrior(nn.Module):

    def __init__(self, lags=1, latent_size=1024, bias=False):
        super().__init__()
        # self.init_hiddens = nn.Parameter(0.001 * torch.randn(lags, latent_size))    
        # out[:,:,0] = (x[:,:,0]@conv.weight[:,:,0].T)+(x[:,:,1]@conv.weight[:,:,1].T) 
        # out[:,:,1] = (x[:,:,1]@conv.weight[:,:,0].T)+(x[:,:,2]@conv.weight[:,:,1].T)
        self.L = lags      
        self.transition = GroupLinearLayer(din = latent_size, 
                                           dout = latent_size, 
                                           num_blocks = lags,
                                           diagonal = False)
        self.bias = bias
        if bias:
            self.b = nn.Parameter(0.001 * torch.randn(1, latent_size))
    
    def forward(self, x, mask=None):
        # x: [BS, T, D] -> [BS, T-L, L+1, D]
        batch_size, length, input_dim = x.shape
        # init_hiddens = self.init_hiddens.repeat(batch_size, 1, 1)
        # x = torch.cat((init_hiddens, x), dim=1)
        x = x.unfold(dimension = 1, size = self.L+1, step = 1)
        x = torch.swapaxes(x, 2, 3)
        shape = x.shape

        x = x.reshape(-1, self.L+1, input_dim)
        xx, yy = x[:,-1:], x[:,:-1]
        if self.bias:
            residuals = torch.sum(self.transition(yy), dim=1) + self.b - xx.squeeze()
        else:
            residuals = torch.sum(self.transition(yy), dim=1) - xx.squeeze()
        residuals = residuals.reshape(batch_size, -1, input_dim)
        # Dummy jacobian matrix (0) to represent identity mapping
        log_abs_det_jacobian = torch.zeros(batch_size, device=x.device)
        return residuals, log_abs_det_jacobian



class ConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ConvGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.W_z = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U_z = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_r = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U_r = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_h = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U_h = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, input, hidden=None):
        # input: B, L, D
        if hidden is None:
            hidden = torch.zeros(input.size(0), self.hidden_dim, device=input.device)
        
        outputs = []
        for t in range(input.size(1)):
            x_t = input[:, t, :]
            z_t = self.sigmoid(self.W_z(x_t) + self.U_z(hidden))
            r_t = self.sigmoid(self.W_r(x_t) + self.U_r(hidden))
            h_tilde = self.tanh(self.W_h(x_t) + self.U_h(r_t * hidden))
            hidden = (1 - z_t) * h_tilde + z_t * hidden
            outputs.append(hidden)
        

        outputs = torch.stack(outputs, dim=1)
        
        return outputs, hidden

class NLayerLeakyMLP(nn.Module):

    def __init__(self, input_dim, output_dim, num_layers, hidden_dim, bias=True):
        super().__init__()
        layers = [ ]
        for l in range(num_layers):
            if l == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
                # layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                # layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: B, L, D
        return self.net(x) # B, L, D


class causal_process(nn.Module):
    def __init__(self, input_dim=2048, output_dim=2048, hidden_dim=512, num_layers=4):
        super().__init__()
        self.lag = 1
        self.z_dim = 1024
        self.register_buffer('base_dist_mean', torch.zeros(self.z_dim))
        self.register_buffer('base_dist_var', torch.eye(self.z_dim))

        # (self, lag=1, z_dim=1024, num_layers=4, hidden_dim=512)
        self.enc = NLayerLeakyMLP(input_dim, output_dim, num_layers, hidden_dim)
        self.dec = NLayerLeakyMLP(int(input_dim/2), output_dim, num_layers, hidden_dim)
        self.rnn = ConvGRU(input_dim, int(output_dim/2))

        self.b_dist = build_dist()
        self.transition_prior = MBDTransitionPrior()
    def reparameterize(self, mean, logvar, random_sampling=True):
        if random_sampling:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5*logvar)
            z = mean + eps*std
            return z
        else:
            return mean
        
    @property
    def base_dist(self):
        # Noise density function
        return Dist.MultivariateNormal(self.base_dist_mean, self.base_dist_var)
    
    def forward(self, x):
        # x: B, D, L
        x = x.permute(0, 2, 1)
        # x: B, L, D
        B, L, D = x.shape
        ft = self.enc(x)
        output, h = self.rnn(ft)
        zs, mus, logvars = self.cal_mus_vars(output)
        zz = self.dec(zs)

        recon_loss = self.reconstruction_loss(x[:,:self.lag], zz[:,:self.lag], 'gaussian') + \
        (self.reconstruction_loss(x[:,self.lag:], zz[:,self.lag:], 'gaussian'))/(L-self.lag)

        q_dist = Dist.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(zs)
        p_dist = Dist.Normal(torch.zeros_like(mus[:,:self.lag]), torch.ones_like(logvars[:,:self.lag]))
        log_pz_normal = torch.sum(torch.sum(p_dist.log_prob(zs[:,:self.lag]),dim=-1),dim=-1)
        log_qz_normal = torch.sum(torch.sum(log_qz[:,:self.lag],dim=-1),dim=-1)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = kld_normal.mean()

        log_qz_laplace = log_qz[:,self.lag:]
        residuals, logabsdet = self.transition_prior(zs)
        sum_log_abs_det_jacobians =  logabsdet
        log_pz_laplace = torch.sum(self.base_dist.log_prob(residuals), dim=1) + sum_log_abs_det_jacobians
        kld_laplace = (torch.sum(torch.sum(log_qz_laplace,dim=-1),dim=-1) - log_pz_laplace) / (L-self.lag)
        kld_laplace = kld_laplace.mean()
        loss = recon_loss + kld_normal + kld_laplace


        return loss, zz

    def cal_mus_vars(self, x):
        # B, L, D/2
        B, L, D = x.shape
        zs, mus, logvars = [], [], []
        for tau in range(self.lag):
            zs.append(torch.ones((B, self.z_dim), device=x.device))

        for t in range(L):
            mid = torch.cat(zs[-self.lag:], dim=1)
            inputs = torch.cat([mid, x[:,t,:]], dim=1)    
            distributions = self.b_dist(inputs)
            mu = distributions[:, :self.z_dim]
            logvar = distributions[:, self.z_dim:]
            zt = self.reparameterize(mu, logvar)
            zs.append(zt)
            mus.append(mu)
            logvars.append(logvar)
        zs = torch.stack(zs, dim=1)
        zs = zs[:,self.lag:]
        mus = torch.stack(mus, dim=1)
        logvars = torch.stack(logvars, dim=1)
        return zs, mus, logvars
    
    def reconstruction_loss(self, x, zz, distribution):
        batch_size = x.size(0)
        assert batch_size != 0

        if distribution == 'bernoulli':
            recon_loss = F.binary_cross_entropy_with_logits(
                zz, x, size_average=False).div(batch_size)

        elif distribution == 'gaussian':
            recon_loss = F.mse_loss(zz, x, size_average=False).div(batch_size)

        elif distribution == 'sigmoid_gaussian':
            zz = F.sigmoid(zz)
            recon_loss = F.mse_loss(zz, x, size_average=False).div(batch_size)

        return recon_loss
