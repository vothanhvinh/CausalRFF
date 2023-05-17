# -*- coding: utf-8 -*-
import numpy as np
import torch
from model import *
from evaluation import Evaluation

FLAGS_NO_TRANSFER = 0
FLAGS_FULL_TRANSFER = 1
FLAGS_LEARN_TRANSFER = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#=============================================================================================================================================
def draw_spectral_SE(d, D):
  return torch.randn((d,D))

def trainW(train_x, train_w, n_sources, source_ranges, D=100, training_iter=200, learning_rate=1e-3, reg=1e-3,
           transfer_flag=FLAGS_LEARN_TRANSFER, display_per_iters=100):
  
  # Generate omega, shared parameters for SE kernel
  omega = draw_spectral_SE(d=train_x.shape[1], D=D)

  # Create models
  model_server = ModelW(x=torch.ones((1,train_x.shape[1])),
                        w=torch.ones((1,1)),
                        n_sources=n_sources, omega=omega, idx_source=0, D=D,
                        transfer_flag=transfer_flag).to(device)

  model_sources = [ModelW(x=train_x[range(source_ranges[idx][0], source_ranges[idx][1]),:],
                          w=train_w[range(source_ranges[idx][0], source_ranges[idx][1])],
                          n_sources=n_sources, omega=omega, idx_source=idx, D=D,
                          transfer_flag=transfer_flag).to(device) for idx in range(n_sources)]
  # Create optimizers
  optimizer_server = torch.optim.Adam(model_server.parameters(), lr=learning_rate)
  optimizer_sources = [torch.optim.Adam(model_sources[idx].parameters(), lr=learning_rate) for idx in range(n_sources)]

  for i in range(training_iter):
    # Compute gradients on each source
    for idx in range(n_sources):
      loss_source = model_sources[idx].loss(reg=reg)
      optimizer_sources[idx].zero_grad()
      loss_source.backward()

      if (i+1)%display_per_iters==0:
        print('Source %d, Iter %d/%d - Loss: %.3f' % (idx, i + 1, training_iter, loss_source.item()))
    
    # Update gradient on server
    loss_server = model_server.loss(reg=reg)
    optimizer_server.zero_grad()
    loss_server.backward() # The purpuse of this line is to allocate memory for gradient of each parameter, i.e param.grad as below
    
    # The previous command optimizer_server.zero_grad() was replaced by following 'for loop' to make it work for the new version of pytorch
    for key, param in model_server.named_parameters():  
      param.grad.zero_()

    for idx in range(n_sources):
      grad_dict_source = {key:param.grad for key, param in model_sources[idx].named_parameters()} # store gradients to grad_dict_source
      
      for key, param in model_server.named_parameters():
        if (param.grad is not None) and param.requires_grad:
          param.grad += grad_dict_source[key]
    optimizer_server.step()

    # Get new model from server
    param_dict_server = {key:param.data for key, param in model_server.named_parameters()}

    # Update new model to all sources
    for idx in range(n_sources):
      for key, param in model_sources[idx].named_parameters():
        if param.requires_grad and param_dict_server[key] is not None:
          param.data = param_dict_server[key] + 0 # + 0 will copy content of param_dict_server[key] to param.data; otherwise, it only assign reference

  return model_server, model_sources, omega

def testW(model_sources, test_x, test_w, n_sources, source_ranges, idx_sources_to_test=None):
  
  if idx_sources_to_test==None:
    idx_lst = range(n_sources)
  else:
    idx_lst = idx_sources_to_test

  wpred_logit = []
  for idx in idx_lst:
    pred = model_sources[idx].pred(test_x[range(source_ranges[idx][0], source_ranges[idx][1]),:]).reshape(-1)
    wpred_logit.append(pred)
  wpred_logit = torch.cat(wpred_logit)

  idx_test = np.concatenate([list(range(source_ranges[idx][0], source_ranges[idx][1])) for idx in idx_lst])
  accur = torch.mean(((wpred_logit>0)*1.0==test_w[idx_test].reshape(-1))*1.0)
  test_stats = np.insert(model_sources[0].trans_factor().cpu().detach().numpy(),0,accur.cpu().detach().numpy())
  return test_stats

def sampleW(model_sources, x, n_sources, source_ranges, n_samples, idx_sources_to_test=None):
  
  if idx_sources_to_test==None:
    idx_lst = range(n_sources)
  else:
    idx_lst = idx_sources_to_test

  w_samples = []
  for idx in idx_lst:
    sample = model_sources[idx].sample(x[range(source_ranges[idx][0], source_ranges[idx][1]),:], n_samples=n_samples)
    w_samples.append(sample)
  
  w_samples = torch.cat(w_samples, dim=0)

  return w_samples

#=============================================================================================================================================
def trainY(train_x, train_y, train_w, n_sources, source_ranges, D=100, is_binary=False,
           training_iter=200, learning_rate=1e-3, reg_beta=1e-3, reg_sig=1e-3,
           transfer_flag=FLAGS_LEARN_TRANSFER, display_per_iters=100):
  
  # Generate omega, shared parameters for SE kernel
  omega = draw_spectral_SE(d=train_x.shape[1], D=D)

  # Create models
  model_server = ModelY(x=torch.ones((1,train_x.shape[1])),
                        y=torch.ones((1,1)),
                        w=torch.ones((1,1)),
                        n_sources=n_sources, omega=omega, idx_source=0, D=D,
                        transfer_flag=transfer_flag).to(device)

  model_sources = [ModelY(x=train_x[range(source_ranges[idx][0], source_ranges[idx][1]),:],
                          y=train_y[range(source_ranges[idx][0], source_ranges[idx][1])],
                          w=train_w[range(source_ranges[idx][0], source_ranges[idx][1])],
                          n_sources=n_sources, omega=omega, idx_source=idx, D=D,
                          transfer_flag=transfer_flag).to(device) for idx in range(n_sources)]
  # Create optimizers
  optimizer_server = torch.optim.Adam(model_server.parameters(), lr=learning_rate)
  optimizer_sources = [torch.optim.Adam(model_sources[idx].parameters(), lr=learning_rate) for idx in range(n_sources)]

  for i in range(training_iter):
    # Compute gradients on each source
    for idx in range(n_sources):
      loss_source = model_sources[idx].loss(reg_beta=reg_beta,reg_sig=reg_sig, is_binary=is_binary)
      optimizer_sources[idx].zero_grad()
      loss_source.backward()

      if (i+1)%display_per_iters==0:
        print('Source %d, Iter %d/%d - Loss: %.3f' % (idx, i + 1, training_iter, loss_source.item()))
    
    # Update gradient on server
    loss_server = model_server.loss(reg_beta=reg_beta,reg_sig=reg_sig, is_binary=is_binary)
    optimizer_server.zero_grad()
    loss_server.backward() # The purpuse of this line is to allocate memory for gradient of each parameter, i.e param.grad as below
    
    # The previous command optimizer_server.zero_grad() was replaced by following 'for loop' to make it work for the new version of pytorch
    for key, param in model_server.named_parameters():  
      param.grad.zero_()
      
    for idx in range(n_sources):
      grad_dict_source = {key:param.grad for key, param in model_sources[idx].named_parameters()} # store gradients to grad_dict_source
      
      for key, param in model_server.named_parameters():
        if (param.grad is not None) and param.requires_grad:
          param.grad += grad_dict_source[key]
    optimizer_server.step()

    # Get new model from server
    param_dict_server = {key:param.data for key, param in model_server.named_parameters()}

    # Update new model to all sources
    for idx in range(n_sources):
      for key, param in model_sources[idx].named_parameters():
        if param.requires_grad and param_dict_server[key] is not None:
          param.data = param_dict_server[key] + 0 # + 0 will copy content of param_dict_server[key] to param.data; otherwise, it only assign reference

  return model_server, model_sources, omega

def testY(model_sources, test_x, test_y, test_w, n_sources, source_ranges, idx_sources_to_test=None):
  
  if idx_sources_to_test==None:
    idx_lst = range(n_sources)
  else:
    idx_lst = idx_sources_to_test

  ypred_logit = []
  for idx in idx_lst:
    pred = model_sources[idx].pred(test_x[range(source_ranges[idx][0], source_ranges[idx][1]),:],
                                   test_w[range(source_ranges[idx][0], source_ranges[idx][1])]).reshape(-1)
    ypred_logit.append(pred)
  ypred_logit = torch.cat(ypred_logit)

  idx_test = np.concatenate([list(range(source_ranges[idx][0], source_ranges[idx][1])) for idx in idx_lst])
  mae = torch.mean(torch.abs(ypred_logit - test_y[idx_test].reshape(-1)))
  rmse = torch.sqrt(torch.mean((ypred_logit - test_y[idx_test].reshape(-1))**2))
  test_stats = np.concatenate((np.asarray([mae.cpu().detach().numpy(), rmse.cpu().detach().numpy()]),
                               model_sources[0].trans_factor().cpu().detach().numpy()))
  return test_stats

def sampleY(model_sources, x, w_samples, n_sources, source_ranges, n_samples, idx_sources_to_test=None, is_binary=False):
  
  if idx_sources_to_test==None:
    idx_lst = range(n_sources)
  else:
    idx_lst = idx_sources_to_test

  y_samples = []
  for idx in idx_lst:
    sample = model_sources[idx].sample(x[range(source_ranges[idx][0], source_ranges[idx][1]),:],
                                       w_samples[range(source_ranges[idx][0], source_ranges[idx][1])],
                                       n_samples=n_samples, is_binary=is_binary)
    y_samples.append(sample)
  y_samples = torch.cat(y_samples, dim=0)
  return y_samples

#=============================================================================================================================================

def trainZY(train_x, train_y, train_w, n_sources, source_ranges, feats_binary=None, feats_continuous=None,
            is_binary=False, dim_z=16, D=100, training_iter=5000, display_per_iters=100,
            transfer_flag=FLAGS_LEARN_TRANSFER, reg_beta=1e-1, learning_rate=1e-2):
  
  # Generate omega, shared parameters for SE kernel
  omega_z = draw_spectral_SE(d=dim_z, D=D)
  omega_xy = draw_spectral_SE(d=train_x.shape[1]+1, D=D)

  # Create models
  model_server = ModelZY(x=torch.ones((1,train_x.shape[1])),
                         y=torch.ones((1,1)),
                         w=torch.ones((1,1)),
                         omega_z=omega_z, omega_xy=omega_xy,
                         n_sources=n_sources, dim_z=dim_z,
                         idx_source=0, D=D,
                         feats_binary=feats_binary,
                         feats_continuous=feats_continuous,
                         transfer_flag=transfer_flag).to(device)

  model_sources = [ModelZY(x=train_x[range(source_ranges[idx][0], source_ranges[idx][1]),:],
                           y=train_y[range(source_ranges[idx][0], source_ranges[idx][1])],
                           w=train_w[range(source_ranges[idx][0], source_ranges[idx][1])],
                           omega_z=omega_z, omega_xy=omega_xy,
                           n_sources=n_sources, dim_z=dim_z,
                           idx_source=idx, D=D,
                           feats_binary=feats_binary,
                           feats_continuous=feats_continuous,
                           transfer_flag=transfer_flag).to(device) for idx in range(n_sources)]

  # Create optimizers
  optimizer_server = torch.optim.Adam(model_server.parameters(), lr=learning_rate)
  optimizer_sources = [torch.optim.Adam(model_sources[idx].parameters(), lr=learning_rate) for idx in range(n_sources)]


  for i in range(training_iter):
    # Compute gradients on each source
    for idx in range(n_sources):
      loss_source = model_sources[idx].loss(reg_beta=reg_beta, is_binary=is_binary)
      optimizer_sources[idx].zero_grad()
      loss_source.backward()

      if (i+1)%display_per_iters==0:
        print('Source %d, Iter %d/%d - Loss: %.3f' % (idx, i + 1, training_iter, loss_source.item()))
    
    # Update gradient on server
    loss_server = model_server.loss(reg_beta=reg_beta, is_binary=is_binary)
    optimizer_server.zero_grad()
    loss_server.backward() # The purpuse of this line is to allocate memory for gradient of each parameter, i.e param.grad as below
    
    # The previous command optimizer_server.zero_grad() was replaced by following 'for loop' to make it work for the new version of pytorch
    for key, param in model_server.named_parameters():  
      param.grad.zero_()
      
    for idx in range(n_sources):
      grad_dict_source = {key:param.grad for key, param in model_sources[idx].named_parameters()} # store gradients to grad_dict_source
      
      for key, param in model_server.named_parameters():
        if (param.grad is not None) and param.requires_grad:
          param.grad += grad_dict_source[key]
    optimizer_server.step()

    # Get new model from server
    param_dict_server = {key:param.data for key, param in model_server.named_parameters()}

    # Update new model to all sources
    for idx in range(n_sources):
      for key, param in model_sources[idx].named_parameters():
        if param.requires_grad and param_dict_server[key] is not None:
          param.data = param_dict_server[key] + 0 # + 0 will copy content of param_dict_server[key] to param.data; otherwise, it only assign reference

  return model_server, model_sources, omega_z, omega_xy

def doW(model_sources, x, w_samples, y_samples, do_w, n_sources, source_ranges, n_samples, idx_sources_to_test=None, is_binary=False, use_mh=False):
  if idx_sources_to_test==None:
    idx_lst = range(n_sources)
  else:
    idx_lst = idx_sources_to_test

  y_mean_do_w = []
  for idx in idx_lst:
    y_mean, _ = model_sources[idx].sample_v2(x=x[range(source_ranges[idx][0], source_ranges[idx][1]),:],
                                             w_samples=w_samples[range(source_ranges[idx][0], source_ranges[idx][1])],
                                             y_samples=y_samples[range(source_ranges[idx][0], source_ranges[idx][1])],
                                             do_w=do_w[range(source_ranges[idx][0], source_ranges[idx][1])],
                                             n_samples=n_samples, is_binary=is_binary, use_mh=use_mh)
    y_mean_do_w.append(y_mean)
  y_mean_do_w = torch.cat(y_mean_do_w, dim=0)
  return y_mean_do_w

def testTEs(model_sources, xte, wte, yte, y_cfte, mute, w_samples, y_samples,
            n_sources, source_ranges, n_samples, idx_sources_to_test=None, is_binary=False, use_mh=False):
  if idx_sources_to_test==None:
    idx_lst = range(n_sources)
  else:
    idx_lst = idx_sources_to_test

  y_samples_do0=doW(model_sources=model_sources, x=xte, w_samples=w_samples, y_samples=y_samples,
                    do_w=torch.zeros((xte.shape[0],1),device=device), n_sources=n_sources,
                    source_ranges=source_ranges, n_samples=n_samples,
                    idx_sources_to_test=idx_sources_to_test, is_binary=is_binary, use_mh=use_mh)
  y_samples_do1=doW(model_sources=model_sources, x=xte, w_samples=w_samples, y_samples=y_samples,
                    do_w=torch.ones((xte.shape[0],1),device=device), n_sources=n_sources,
                    source_ranges=source_ranges, n_samples=n_samples,
                    idx_sources_to_test=idx_sources_to_test, is_binary=is_binary, use_mh=use_mh)

  y_do0_mean = torch.mean(y_samples_do0,dim=1)
  y_do1_mean = torch.mean(y_samples_do1,dim=1)

  y_mean_do_w = []
  y0te = []
  y1te = []
  mute_ = []
  for idx in idx_lst:
    w = wte[range(source_ranges[idx][0], source_ranges[idx][1]),:]
    y = yte[range(source_ranges[idx][0], source_ranges[idx][1]),:]
    y_cf = y_cfte[range(source_ranges[idx][0], source_ranges[idx][1]),:]
    mu = mute[range(source_ranges[idx][0], source_ranges[idx][1]),:]
    y0 = (1-w)*y + w*y_cf
    y1 = w*y + (1-w)*y_cf
    y0te.append(y0)
    y1te.append(y1)
    mute_.append(mu)

  y0te = torch.cat(y0te,dim=0).reshape(-1)
  y1te = torch.cat(y1te,dim=0).reshape(-1)
  mute_ = np.concatenate(mute_,axis=0)
  print(np.mean(-mute_[:,0] + mute_[:,1]))
  print(np.mean(-y_do0_mean.cpu().detach().numpy() + y_do1_mean.cpu().detach().numpy()))

  eval = Evaluation(mute_[:,0], mute_[:,1], y0te.cpu().detach().numpy(), y1te.cpu().detach().numpy())
  abs_err = eval.absolute_err_ate(y_do0_mean.cpu().detach().numpy(), y_do1_mean.cpu().detach().numpy())
  pehe = eval.pehe(y_do0_mean.cpu().detach().numpy(), y_do1_mean.cpu().detach().numpy())
  return abs_err, pehe
