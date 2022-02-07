# -*- coding: utf-8 -*-
import torch
import numpy as np
FLAGS_NO_TRANSFER = 0
FLAGS_FULL_TRANSFER = 1
FLAGS_LEARN_TRANSFER = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#=============================================================================================================================================
class ModelW(torch.nn.Module):
  def __init__(self, x, w, n_sources, omega, idx_source=-1, D=100, transfer_flag=FLAGS_LEARN_TRANSFER):
    super().__init__()
    self.loss_bce = torch.nn.BCEWithLogitsLoss(reduction='sum')
    self.sigmoid = torch.nn.Sigmoid()

    self.n_sources = n_sources
    self.omega = omega
    self.D = D
    self.x = x
    self.w = w
    self.transfer_flag = transfer_flag
    self.idx_source = idx_source

    # Transfer factors
    self.transfer_factor_logit = torch.nn.Parameter(torch.zeros(int(((self.n_sources-1)*self.n_sources)/2)))
    self.transfer_factor_logit_idx_mapping = self.create_trans_factor_idx_mapping()
    self.transfer_factor_idx_less, self.transfer_factor_idx_greater = self.get_trans_factor_idx()

    # Parameters
    self.beta = torch.nn.Parameter(torch.rand((2*self.D,self.n_sources)))

    # Parameter for SE kernel
    self.ell_logit = torch.nn.Parameter(torch.tensor(0.0))

  def phi_x(self, x):
    return torch.cat([torch.cos(x.mm(torch.exp(self.ell_logit)*self.omega)),
                      torch.sin(x.mm(torch.exp(self.ell_logit)*self.omega))],axis=1)/np.sqrt(self.D)

  def create_trans_factor_idx_mapping(self):
    transfer_factor_logit_idx_mapping = {}
    k = 0
    for i in range(self.n_sources):
      for j in range(i+1,self.n_sources):
        transfer_factor_logit_idx_mapping[str(i)+str(j)] = k
        k += 1
    return transfer_factor_logit_idx_mapping

  def get_trans_factor_idx(self):
    idx_less = []
    idx_greater = []
    for i in range(self.n_sources):
      if i < self.idx_source:
        idx_less.append(self.transfer_factor_logit_idx_mapping[str(i)+str(self.idx_source)])
      elif i > self.idx_source:
        idx_greater.append(self.transfer_factor_logit_idx_mapping[str(self.idx_source)+str(i)])
    return idx_less, idx_greater

  def forward(self, x_Fourier, idx_source=None):
    if idx_source==None:
      idx_source = self.idx_source

    if self.transfer_flag == FLAGS_FULL_TRANSFER:
      weights = torch.sum(self.beta,axis=1,keepdim=True)
    elif self.transfer_flag == FLAGS_NO_TRANSFER:
      weights = self.beta[:,idx_source:idx_source+1]
    elif self.transfer_flag == FLAGS_LEARN_TRANSFER:
      transfer_factor_less = self.sigmoid(self.transfer_factor_logit[self.transfer_factor_idx_less])
      transfer_factor_greater = self.sigmoid(self.transfer_factor_logit[self.transfer_factor_idx_greater])
      weights = torch.sum(self.beta[:,:idx_source]*transfer_factor_less,axis=1,keepdim=True) + \
                self.beta[:,idx_source:idx_source+1] + \
                torch.sum(self.beta[:,idx_source+1:]*transfer_factor_greater,axis=1,keepdim=True)

    return (x_Fourier.matmul(weights)).reshape(-1), weights

  def trans_factor(self):
    return self.sigmoid(self.transfer_factor_logit)

  def pred(self, x, idx_source=None):
    if idx_source==None:
      idx_source = self.idx_source
    f_preds,_ = self.forward(self.phi_x(x), idx_source=idx_source)
    return f_preds

  def sample(self, x, n_samples, idx_source=None):
    if idx_source==None:
      idx_source = self.idx_source

    f_preds,_ = self.forward(self.phi_x(x), idx_source=idx_source)
    m = torch.distributions.bernoulli.Bernoulli(logits=f_preds.reshape(-1))
    w_samples = m.sample((n_samples,)).t()
    return w_samples

  def loss(self, reg=1e-3):
    ypred_logit, weights = self.forward(self.phi_x(self.x))
    regularizer = reg*torch.sum(self.beta**2)
    # regularizer = reg*torch.sum(weights**2)
    return self.loss_bce(ypred_logit.reshape(-1), self.w.reshape(-1)) + regularizer/self.n_sources

#=============================================================================================================================================
class ModelY(torch.nn.Module):
  def __init__(self, x, y, w, n_sources, omega, idx_source=-1, D=100, transfer_flag=FLAGS_LEARN_TRANSFER):
    super().__init__()
    self.loss_bce = torch.nn.BCEWithLogitsLoss(reduction='sum')
    self.loss_mse = torch.nn.MSELoss(reduction='sum')
    self.sigmoid = torch.nn.Sigmoid()

    self.n_sources = n_sources
    self.omega = omega
    self.D = D
    self.x = x
    self.y = y
    self.w = w 
    self.transfer_flag = transfer_flag
    self.idx_source = idx_source

    # Transfer factors
    self.transfer_factor_logit = torch.nn.Parameter(torch.zeros(int(((self.n_sources-1)*self.n_sources)/2)))
    self.transfer_factor_logit_idx_mapping = self.create_trans_factor_idx_mapping()
    self.transfer_factor_idx_less, self.transfer_factor_idx_greater = self.get_trans_factor_idx()

    # Parameters
    self.beta0 = torch.nn.Parameter(torch.rand((2*self.D,self.n_sources)))
    self.beta1 = torch.nn.Parameter(torch.rand((2*self.D,self.n_sources)))
    self.sigma_y_logit = torch.nn.Parameter(torch.tensor(0.0))

    # Parameter for SE kernel
    self.ell_logit = torch.nn.Parameter(torch.tensor(0.0))

  def phi_x(self, x):
    return torch.cat([torch.cos(x.mm(torch.exp(self.ell_logit)*self.omega)),
                      torch.sin(x.mm(torch.exp(self.ell_logit)*self.omega))],axis=1)/np.sqrt(self.D)

  def create_trans_factor_idx_mapping(self):
    transfer_factor_logit_idx_mapping = {}
    k = 0
    for i in range(self.n_sources):
      for j in range(i+1,self.n_sources):
        transfer_factor_logit_idx_mapping[str(i)+str(j)] = k
        k += 1
    return transfer_factor_logit_idx_mapping

  def get_trans_factor_idx(self):
    idx_less = []
    idx_greater = []
    for i in range(self.n_sources):
      if i < self.idx_source:
        idx_less.append(self.transfer_factor_logit_idx_mapping[str(i)+str(self.idx_source)])
      elif i > self.idx_source:
        idx_greater.append(self.transfer_factor_logit_idx_mapping[str(self.idx_source)+str(i)])
    return idx_less, idx_greater

  def forward(self, x_Fourier, w, idx_source=None):
    if idx_source==None:
      idx_source = self.idx_source

    if self.transfer_flag == FLAGS_FULL_TRANSFER:
      weights0 = torch.sum(self.beta0,axis=1,keepdim=True)
      weights1 = torch.sum(self.beta1,axis=1,keepdim=True)
    elif self.transfer_flag == FLAGS_NO_TRANSFER:
      weights0 = self.beta0[:,idx_source:idx_source+1]
      weights1 = self.beta1[:,idx_source:idx_source+1]
    elif self.transfer_flag == FLAGS_LEARN_TRANSFER:
      transfer_factor_less = self.sigmoid(self.transfer_factor_logit[self.transfer_factor_idx_less])
      transfer_factor_greater = self.sigmoid(self.transfer_factor_logit[self.transfer_factor_idx_greater])
      weights0 = torch.sum(self.beta0[:,:idx_source]*transfer_factor_less,axis=1,keepdim=True) + \
                self.beta0[:,idx_source:idx_source+1] + \
                torch.sum(self.beta0[:,idx_source+1:]*transfer_factor_greater,axis=1,keepdim=True)
      weights1 = torch.sum(self.beta1[:,:idx_source]*transfer_factor_less,axis=1,keepdim=True) + \
                self.beta1[:,idx_source:idx_source+1] + \
                torch.sum(self.beta1[:,idx_source+1:]*transfer_factor_greater,axis=1,keepdim=True)

    return (1-w).reshape(-1)*(x_Fourier.matmul(weights0)).reshape(-1) + w.reshape(-1)*(x_Fourier.matmul(weights1)).reshape(-1), weights0, weights1

  def trans_factor(self):
    return self.sigmoid(self.transfer_factor_logit)

  def pred(self, x, w, idx_source=None):
    if idx_source==None:
      idx_source = self.idx_source

    f_preds,_,_ = self.forward(self.phi_x(x), w, idx_source=idx_source)
    return f_preds

  def sample(self, x, w_samples, n_samples, idx_source=None, is_binary=False):
    if idx_source==None:
      idx_source = self.idx_source

    y_samples = []
    for i in range(n_samples):
      f_preds,_,_ = self.forward(self.phi_x(x), w_samples[:,i:i+1], idx_source=idx_source)
      if is_binary == True:
        m = torch.distributions.bernoulli.Bernoulli(logits=f_preds.reshape(-1))
      else:
        m = torch.distributions.normal.Normal(f_preds.reshape(-1), torch.exp(self.sigma_y_logit))
      y_sample = m.sample((1,)).t()
      y_samples.append(y_sample)
    return torch.cat(y_samples, dim=1)

  def loss(self, reg_beta=1e-3, reg_sig=1e-3, is_binary=False):
    ypred_logit, weights0, weights1 = self.forward(self.phi_x(self.x), self.w)
    reg = reg_beta*torch.sum(self.beta0**2)/self.n_sources + reg_beta*torch.sum(self.beta1**2)/self.n_sources
    if is_binary == True:
      return self.loss_bce(ypred_logit.reshape(-1), self.y.reshape(-1)) + reg
    else:
      reg = (reg + reg_sig*self.sigma_y_logit**2)/self.n_sources
      return self.x.shape[0]*self.sigma_y_logit + (0.5/torch.exp(2*self.sigma_y_logit))*self.loss_mse(ypred_logit.reshape(-1), self.y.reshape(-1)) + reg
    
#=============================================================================================================================================
class ModelZY(torch.nn.Module):
  def __init__(self, x, y, w, omega_z, omega_xy, n_sources, dim_z=16, idx_source=-1, D=100,
               feats_binary=None, feats_continuous=None, transfer_flag=FLAGS_LEARN_TRANSFER):
    super().__init__()
    self.loss_bce = torch.nn.BCEWithLogitsLoss(reduction='sum')
    self.loss_mse = torch.nn.MSELoss(reduction='sum')
    self.sigmoid = torch.nn.Sigmoid()
    self.logsigmoid = torch.nn.LogSigmoid()

    self.n_sources = n_sources
    self.omega_z = omega_z
    self.omega_xy = omega_xy
    self.D = D
    self.x = x
    self.y = y
    self.w = w
    self.dim_z = dim_z
    self.transfer_flag = transfer_flag
    self.idx_source = idx_source
    self.feats_binary = feats_binary
    self.feats_continuous = feats_continuous

    # Transfer factors
    self.transfer_factor_logit = torch.nn.Parameter(torch.zeros(int(((self.n_sources-1)*self.n_sources)/2)))
    self.transfer_factor_logit_idx_mapping = self.create_trans_factor_idx_mapping()
    self.transfer_factor_idx_less, self.transfer_factor_idx_greater = self.get_trans_factor_idx()

    # Parameters
    self.betaz0 = torch.nn.Parameter(torch.rand((2*self.D,dim_z,self.n_sources)))
    self.betaz1 = torch.nn.Parameter(torch.rand((2*self.D,dim_z,self.n_sources)))
    self.sigma_z_logit = torch.nn.Parameter(torch.rand(dim_z))

    self.betay0 = torch.nn.Parameter(torch.rand((2*self.D,self.n_sources)))
    self.betay1 = torch.nn.Parameter(torch.rand((2*self.D,self.n_sources)))
    self.sigma_y_logit = torch.nn.Parameter(torch.tensor(0.0))

    self.betaw = torch.nn.Parameter(torch.rand((2*self.D,self.n_sources)))
    self.betax = torch.nn.Parameter(torch.rand((2*self.D,x.shape[1],self.n_sources)))

    # Parameter for SE kernel
    self.ell_logit = torch.nn.Parameter(torch.tensor(0.0))
    # self.ell_logit_z = torch.nn.Parameter(torch.tensor(0.0))
    self.debug = False

  def phi_xy(self, xy):
    return torch.cat([torch.cos(xy.mm(torch.exp(self.ell_logit)*self.omega_xy)),
                      torch.sin(xy.mm(torch.exp(self.ell_logit)*self.omega_xy))],axis=1)/np.sqrt(self.D)
  def phi_z(self, z):
    return torch.cat([torch.cos(z.mm(torch.exp(self.ell_logit)*self.omega_z)),
                      torch.sin(z.mm(torch.exp(self.ell_logit)*self.omega_z))],axis=1)/np.sqrt(self.D)

  def create_trans_factor_idx_mapping(self):
    transfer_factor_logit_idx_mapping = {}
    k = 0
    for i in range(self.n_sources):
      for j in range(i+1,self.n_sources):
        transfer_factor_logit_idx_mapping[str(i)+str(j)] = k
        k += 1
    return transfer_factor_logit_idx_mapping

  def get_trans_factor_idx(self):
    idx_less = []
    idx_greater = []
    for i in range(self.n_sources):
      if i < self.idx_source:
        idx_less.append(self.transfer_factor_logit_idx_mapping[str(i)+str(self.idx_source)])
      elif i > self.idx_source:
        idx_greater.append(self.transfer_factor_logit_idx_mapping[str(self.idx_source)+str(i)])
    return idx_less, idx_greater

  def fz(self, xy_Fourier, w, idx_source=None):
    if idx_source==None:
      idx_source = self.idx_source

    if self.transfer_flag == FLAGS_FULL_TRANSFER:
      weights0 = torch.sum(self.betaz0,axis=2)
      weights1 = torch.sum(self.betaz1,axis=2)
    elif self.transfer_flag == FLAGS_NO_TRANSFER:
      weights0 = self.betaz0[:,:,idx_source]
      weights1 = self.betaz1[:,:,idx_source]
    elif self.transfer_flag == FLAGS_LEARN_TRANSFER:
      transfer_factor_less = self.sigmoid(self.transfer_factor_logit[self.transfer_factor_idx_less])
      transfer_factor_greater = self.sigmoid(self.transfer_factor_logit[self.transfer_factor_idx_greater])
      weights0 = torch.sum(self.betaz0[:,:,:idx_source]*transfer_factor_less,axis=2) + \
                self.betaz0[:,:,idx_source] + \
                torch.sum(self.betaz0[:,:,idx_source+1:]*transfer_factor_greater,axis=2)
      weights1 = torch.sum(self.betaz1[:,:,:idx_source]*transfer_factor_less,axis=2) + \
                self.betaz1[:,:,idx_source] + \
                torch.sum(self.betaz1[:,:,idx_source+1:]*transfer_factor_greater,axis=2)

    return (1-w)*xy_Fourier.matmul(weights0) + w*xy_Fourier.matmul(weights1), weights0, weights1
  
  def fy(self, z_samples_Fourier, w, idx_source=None):
    if idx_source==None:
      idx_source = self.idx_source

    if self.transfer_flag == FLAGS_FULL_TRANSFER:
      weights0 = torch.sum(self.betay0,axis=1,keepdim=True)
      weights1 = torch.sum(self.betay1,axis=1,keepdim=True)
    elif self.transfer_flag == FLAGS_NO_TRANSFER:
      weights0 = self.betay0[:,idx_source:idx_source+1]
      weights1 = self.betay1[:,idx_source:idx_source+1]
    elif self.transfer_flag == FLAGS_LEARN_TRANSFER:
      transfer_factor_less = self.sigmoid(self.transfer_factor_logit[self.transfer_factor_idx_less])
      transfer_factor_greater = self.sigmoid(self.transfer_factor_logit[self.transfer_factor_idx_greater])

      weights0 = torch.sum(self.betay0[:,:idx_source]*transfer_factor_less,axis=1,keepdim=True) + \
                self.betay0[:,idx_source:idx_source+1] + \
                torch.sum(self.betay0[:,idx_source+1:]*transfer_factor_greater,axis=1,keepdim=True)
      weights1 = torch.sum(self.betay1[:,:idx_source]*transfer_factor_less,axis=1,keepdim=True) + \
                self.betay1[:,idx_source:idx_source+1] + \
                torch.sum(self.betay1[:,idx_source+1:]*transfer_factor_greater,axis=1,keepdim=True)

    return (1-w).reshape(-1)*(z_samples_Fourier.matmul(weights0)).reshape(-1) + w.reshape(-1)*(z_samples_Fourier.matmul(weights1)).reshape(-1), weights0, weights1

  def fw(self, z_samples_Fourier, idx_source=None):
    if idx_source==None:
      idx_source = self.idx_source

    if self.transfer_flag == FLAGS_FULL_TRANSFER:
      weights = torch.sum(self.betaw,axis=1,keepdim=True)
    elif self.transfer_flag == FLAGS_NO_TRANSFER:
      weights = self.betaw[:,idx_source:idx_source+1]
    elif self.transfer_flag == FLAGS_LEARN_TRANSFER:
      transfer_factor_less = self.sigmoid(self.transfer_factor_logit[self.transfer_factor_idx_less])
      transfer_factor_greater = self.sigmoid(self.transfer_factor_logit[self.transfer_factor_idx_greater])
      weights = torch.sum(self.betaw[:,:idx_source]*transfer_factor_less,axis=1,keepdim=True) + \
                self.betaw[:,idx_source:idx_source+1] + \
                torch.sum(self.betaw[:,idx_source+1:]*transfer_factor_greater,axis=1,keepdim=True)

    return (z_samples_Fourier.matmul(weights)).reshape(-1), weights

  def fx(self, z_samples_Fourier, idx_source=None):
    if idx_source==None:
      idx_source = self.idx_source

    if self.transfer_flag == FLAGS_FULL_TRANSFER:
      weights = torch.sum(self.betax,axis=2)
    elif self.transfer_flag == FLAGS_NO_TRANSFER:
      weights = self.betax[:,:,idx_source]
    elif self.transfer_flag == FLAGS_LEARN_TRANSFER:
      transfer_factor_less = self.sigmoid(self.transfer_factor_logit[self.transfer_factor_idx_less])
      transfer_factor_greater = self.sigmoid(self.transfer_factor_logit[self.transfer_factor_idx_greater])
      weights = torch.sum(self.betax[:,:,:idx_source]*transfer_factor_less,axis=2) + \
                self.betax[:,:,idx_source] + \
                torch.sum(self.betax[:,:,idx_source+1:]*transfer_factor_greater,axis=2)

    return z_samples_Fourier.matmul(weights), weights

  def forward(self, x, y, w):
    xy_Fourier = self.phi_xy(xy=torch.cat((x,y),dim=1))
    fz, weightsz0, weightsz1 = self.fz(xy_Fourier, w)
    sigmaz = torch.exp(self.sigma_z_logit)
    z_samples = fz + sigmaz*torch.rand((x.shape[0], self.dim_z), device=device)
    z_samples_Fourier = self.phi_z(z=z_samples)
    fy, weightsy0, weightsy1 = self.fy(z_samples_Fourier, w)
    fw, weightsw = self.fw(z_samples_Fourier)
    fx, weightsx = self.fx(z_samples_Fourier)
    return fz, fy, fw, fx, weightsz0, weightsz1, weightsy0, weightsy1, weightsw, weightsx

  def trans_factor(self):
    return self.sigmoid(self.transfer_factor_logit).detach()

  def pred_z(self, x, y, w):
    xy_Fourier = self.phi_xy(xy=torch.cat((x,y),dim=1))
    f_preds,_,_ = self.fz(xy_Fourier, w)
    return f_preds.detach().clone()

  def pred_y(self, z, w):
    z_Fourier = self.phi_z(z)
    f_preds,_,_ = self.fy(z_Fourier, w)
    return f_preds.detach().clone()

  def sample_z(self, x, y_samples, w_samples, n_samples):
    sigmaz = torch.exp(self.sigma_z_logit)
    z_samples = []
    for i in range(n_samples):
      fz = self.pred_z(x, y_samples[:,i:i+1], w_samples[:,i:i+1])
      z_sample = fz + sigmaz*torch.rand((x.shape[0], self.dim_z), device=device)
      z_samples.append(z_sample)

    return z_samples

  def log_lik_z_mh(self, x, y_sample, w_sample, z_sample, is_binary=False):
    self.debug=True
    # log P(Y|Z,W) (excluded constants)
    pred_y_logit = self.pred_y(z_sample, w_sample)
    if is_binary==False:
      loglik_y = (-0.5*(y_sample.reshape(-1) - pred_y_logit.reshape(-1))**2).reshape(-1,1)
    else:
      loglik_y = (y_sample.reshape(-1)*self.logsigmoid(pred_y_logit.reshape(-1)) \
                  + (1-y_sample.reshape(-1))*self.logsigmoid(-pred_y_logit.reshape(-1))).reshape(-1,1)
    # log P(W|Z)
    z_Fourier = self.phi_z(z_sample).detach().clone()
    pred_w_logit,_ = self.fw(z_Fourier)
    pred_w_logit = pred_w_logit.detach().clone()
    loglik_w = (w_sample.reshape(-1)*self.logsigmoid(pred_w_logit.reshape(-1)) \
                + (1-w_sample.reshape(-1))*self.logsigmoid(-pred_w_logit.reshape(-1))).reshape(-1,1)
    # log P(X|Z)
    pred_x_logit,_ = self.fx(z_Fourier)
    if self.feats_continuous==None and self.feats_binary==None: # default is bce for binary features
        loglik_x = (torch.sum(x*self.logsigmoid(pred_x_logit) + (1-x)*self.logsigmoid(-pred_x_logit),dim=1)).reshape(-1,1)
    elif self.feats_continuous==None: # for binary features
        loglik_x = (torch.sum(x*self.logsigmoid(pred_x_logit) + (1-x)*self.logsigmoid(-pred_x_logit),dim=1)).reshape(-1,1)
    elif self.feats_binary==None: # for continuous features
        loglik_x = (torch.sum(-0.5*(x - pred_x_logit)**2, dim=1)).reshape(-1,1)
    else: # for both binary and continuous features
        loglik_x_bin = (torch.sum(x[:,self.feats_binary]*self.logsigmoid(pred_x_logit[:,self.feats_binary])\
                                  + (1-x[:,self.feats_binary])*self.logsigmoid(-pred_x_logit[:,self.feats_binary]),dim=1)).reshape(-1,1)
        loglik_x_con = (torch.sum(-0.5*(x[:,self.feats_continuous] - pred_x_logit[:,self.feats_continuous])**2, dim=1)).reshape(-1,1)
        loglik_x = loglik_x_bin + loglik_x_con

    # log P(Z)
    loglik_z = (-0.5*torch.sum(z_sample**2,dim=1)).reshape(-1,1)            

    return loglik_y + loglik_w + loglik_x + loglik_z

  def log_q(self, x, y_sample, w_sample, z_sample):
    pred_z_logit = self.pred_z(x=x, y=y_sample, w=w_sample)
    log_q_z = (torch.sum(-0.5*(z_sample - pred_z_logit)**2/(torch.exp(2*self.sigma_z_logit).reshape(-1)), dim=1)).reshape(-1,1)
    return log_q_z
    
  def sample_z_mh(self, x, y_sample, w_sample, z_samples, is_binary=False):
    sample = [z_samples[0]]
    for i in range(len(z_samples)):
      
      logAP = self.log_lik_z_mh(x, y_sample, w_sample, z_samples[i], is_binary) - self.log_lik_z_mh(x, y_sample, w_sample, sample[-1], is_binary)\
                + self.log_q(x, y_sample, w_sample, sample[-1]) - self.log_q(x, y_sample, w_sample, z_samples[i])

      logU = torch.from_numpy(np.log(np.random.uniform(0,1, (x.shape[0],1)))).to(device)
      
      new_sample = torch.zeros((z_samples[0].shape[0], z_samples[0].shape[1]), dtype=torch.float32)
      choice = logU.reshape(-1) < logAP.reshape(-1)
      
      new_sample[choice,:] = z_samples[i][choice,:] # Accept new sample
      new_sample[~choice,:] = sample[-1][~choice,:] # Reject new sample (keep previous sample)
      sample.append(new_sample)

    return sample[-1]

  def sample_z_v2(self, x, y_samples, w_samples, n_samples, is_binary=False): # independent sampler
    sigmaz = torch.exp(self.sigma_z_logit)
    z_samples = []
    for i in range(n_samples):
      fz = self.pred_z(x, y_samples[:,i:i+1], w_samples[:,i:i+1])

      samples = []
      for y in range(100):
        z_sample = fz + sigmaz*torch.rand((x.shape[0], self.dim_z), device=device)
        samples.append(z_sample)
      z_sample = self.sample_z_mh(x, y_samples[:,i:i+1], w_samples[:,i:i+1], samples, is_binary=is_binary)

      z_samples.append(z_sample)

    return z_samples

  def sample_y(self, do_w, z_sample, n_samples, is_binary=False):
    f_preds = self.pred_y(z_sample, do_w)
    if is_binary == True:
      m = torch.distributions.bernoulli.Bernoulli(logits=f_preds)
    else:
      m = torch.distributions.normal.Normal(f_preds, 1)
    y_samples = m.sample((n_samples,)).t()
    return y_samples

  def sample(self, x, do_w, y_samples, w_samples, n_samples, is_binary=False):
    z_samples = self.sample_z(x, y_samples, w_samples, n_samples)
    y_samples = []
    for z_sample in z_samples:
      y_sample = self.sample_y(do_w, z_sample=z_sample, n_samples=1, is_binary=is_binary)
      y_samples.append(y_sample)
    return  torch.cat(y_samples,dim=1), z_samples

  def sample_v2(self, x, w_samples, y_samples, do_w, n_samples, is_binary=False, use_mh=False):
    if use_mh == False:
      z_samples = self.sample_z(x, y_samples, w_samples, n_samples)
    else:
      z_samples = self.sample_z_v2(x, y_samples, w_samples, n_samples, is_binary=is_binary)
    y_samples = []
    for z_sample in z_samples:
      if is_binary==False:
        y_sample = self.pred_y(z_sample, do_w)
      else:
        y_sample = self.sigmoid(self.pred_y(z_sample, do_w))
      y_samples.append(y_sample.reshape(-1,1))

    return  torch.cat(y_samples,dim=1), z_samples

  def loss(self, is_binary=False, reg_beta=0.1):
    zpred, ypred_logit, wpred_logit, xpred, weightsz0, weightsz1, weightsy0, weightsy1, weightsw, weightsx = self.forward(self.x, self.y, self.w)
    KL = 0.5*torch.sum(torch.exp(2*self.sigma_z_logit) - 2*self.sigma_z_logit)*self.x.shape[0] + 0.5*torch.sum(zpred**2)
    reg = reg_beta*torch.sum(self.betaz0**2) + reg_beta*torch.sum(self.betaz1**2) \
          + reg_beta*torch.sum(self.betay0**2) + reg_beta*torch.sum(self.betay1**2) \
          + reg_beta*torch.sum(self.betaw**2) + reg_beta*torch.sum(self.betax**2)

    if self.feats_continuous==None and self.feats_binary==None: # default is bce for binary features
        loss_x = self.loss_bce(xpred, self.x)
    elif self.feats_continuous==None: # for binary features
        loss_x = self.loss_bce(xpred, self.x)
    elif self.feats_binary==None: # for continuous features
        loss_x = self.loss_mse(xpred, self.x)
    else: # for both binary and continuous features
        loss_x = self.loss_bce(xpred[:,self.feats_binary], self.x[:,self.feats_binary]) \
                  + self.loss_mse(xpred[:,self.feats_continuous], self.x[:,self.feats_continuous])
    
    if is_binary == True: # binary outcomes
      return self.loss_bce(ypred_logit.reshape(-1), self.y.reshape(-1)) \
             + self.loss_bce(wpred_logit.reshape(-1), self.w.reshape(-1)) \
             + loss_x\
             + KL + reg/self.n_sources
    else: # continuous outcomes
      return self.loss_mse(ypred_logit.reshape(-1), self.y.reshape(-1)) \
              + self.loss_bce(wpred_logit.reshape(-1), self.w.reshape(-1)) \
              + loss_x\
              + KL + reg/self.n_sources