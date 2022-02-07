# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


# Synthetic dataset multi-source, the same distribution
class SynData5Sources:
  def __init__(self):
    data = np.load('datasets/Synthetic_Data/data-5sources.npz', allow_pickle=True)
    self.data_lst = data['data_lst']
    self.n_replicates = data['n_replicates'].item()
    self.n_sources = data['n_sources'].item()
    self.source_size = data['Ts'].item()
    self.train_size = 30
    self.test_size = 800
    self.val_size = 170
    
  def get_train_test_val(self, m_sources=1):
    for i in range(self.n_replicates):
      data = self.data_lst[i]
      n_data_points = m_sources*self.source_size
      t, y, y_cf = data[:n_data_points, 0][:, np.newaxis], data[:n_data_points, 1][:, np.newaxis], data[:n_data_points, 2][:, np.newaxis]
      mu_0, mu_1, x = data[:n_data_points, 3][:, np.newaxis], data[:n_data_points, 4][:, np.newaxis], data[:n_data_points, 5:]

      idx_train = np.concatenate([list(range(i, i+self.train_size))
                                  for i in np.arange(0, n_data_points, self.source_size)])
      idx_test = np.concatenate([list(range(i+self.train_size, i+self.train_size+self.test_size))
                                  for i in np.arange(0, n_data_points, self.source_size)])
      idx_val = np.concatenate([list(range(i+self.train_size+self.test_size,i+self.train_size+self.test_size+self.val_size))
                                  for i in np.arange(0, n_data_points, self.source_size)])

      train = (x[idx_train], t[idx_train], y[idx_train]), (y_cf[idx_train], mu_0[idx_train], mu_1[idx_train])
      test = (x[idx_test], t[idx_test], y[idx_test]), (y_cf[idx_test], mu_0[idx_test], mu_1[idx_test])
      val = (x[idx_val], t[idx_val], y[idx_val]), (y_cf[idx_val], mu_0[idx_val], mu_1[idx_val])
      yield train, test, val

# Synthetic dataset multi-source, different distribution
class SynData5SourcesDiff:
  def __init__(self):
    data = np.load('datasets/Synthetic_Data/data-5sources-diff-dist.npz', allow_pickle=True)
    self.data_lst = data['data_lst']
    self.n_replicates = data['n_replicates'].item()
    self.n_sources = data['n_sources'].item()
    self.source_size = data['Ts'].item()
    self.train_size = 100
    self.test_size = 800
    self.val_size = 100
    
  def get_train_test_val(self, m_sources=1):
    for i in range(self.n_replicates):
      data = self.data_lst[i]
      n_data_points = m_sources*self.source_size
      t, y, y_cf = data[:n_data_points, 0][:, np.newaxis], data[:n_data_points, 1][:, np.newaxis], data[:n_data_points, 2][:, np.newaxis]
      mu_0, mu_1, x = data[:n_data_points, 3][:, np.newaxis], data[:n_data_points, 4][:, np.newaxis], data[:n_data_points, 5:]

      idx_train = np.concatenate([list(range(i, i+self.train_size))
                                  for i in np.arange(0, n_data_points, self.source_size)])
      idx_test = np.concatenate([list(range(i+self.train_size, i+self.train_size+self.test_size))
                                  for i in np.arange(0, n_data_points, self.source_size)])
      idx_val = np.concatenate([list(range(i+self.train_size+self.test_size,i+self.train_size+self.test_size+self.val_size))
                                  for i in np.arange(0, n_data_points, self.source_size)])

      train = (x[idx_train], t[idx_train], y[idx_train]), (y_cf[idx_train], mu_0[idx_train], mu_1[idx_train])
      test = (x[idx_test], t[idx_test], y[idx_test]), (y_cf[idx_test], mu_0[idx_test], mu_1[idx_test])
      val = (x[idx_val], t[idx_val], y[idx_val]), (y_cf[idx_val], mu_0[idx_val], mu_1[idx_val])
      yield train, test, val


# IHDP dataset
class IHDP(object):
  def __init__(self,use_one_hot=False):
    self.n_sources = 3
    self.source_size = 249
    self.n_replicates = 8
    self.train_size = 50
    self.test_size = 100
    self.val_size = 99
    self.use_one_hot = use_one_hot

    # which features are binary
    self.binfeats = list(range(6,25))
    # which features are continuous
    self.contfeats = list(range(0,6))

  def get_train_test_val(self, m_sources):
    for i in range(self.n_replicates):
      data = pd.read_csv('datasets/IHDP/csv/ihdp_npci_{}.csv'.format(i+1),header=None).values
      # data = self.data_lst_Delta[0][i]

      t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
      mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]

      if self.use_one_hot==True and m_sources > 0:
        one_hot = pd.get_dummies(np.concatenate([[i]*self.source_size for i in range(self.n_sources)])).values
        one_hot = one_hot[:,:m_sources]
        x = np.concatenate((x,one_hot),axis=1)
        print('Use one-hot encoding, d_x = {}'.format(x.shape[1]))
      else:
        print('Do not use one-hot encoding, d_x = {}'.format(x.shape[1]))

      # itr = list(range(0,self.Ts*m_sources)) + list(range(self.Ts*self.n_sources,self.Ts*self.n_sources+50))
      # ite = list(range(self.Ts*self.n_sources+50,self.T-n_val))
      # iva = list(range(self.T-n_val,self.T))
      itr = np.concatenate([range(i,i+self.train_size)
                                    for i in range(0, m_sources*self.source_size, self.source_size)])
      ite = np.concatenate([range(i+self.train_size,i+self.train_size+self.test_size)
                                    for i in range(0, m_sources*self.source_size, self.source_size)])
      iva = np.concatenate([range(i+self.train_size+self.test_size,i+self.train_size+self.test_size+self.val_size)
                                    for i in range(0, m_sources*self.source_size, self.source_size)])
      train = (x[itr], t[itr], y[itr]), (y_cf[itr], mu_0[itr], mu_1[itr])
      valid = (x[iva], t[iva], y[iva]), (y_cf[iva], mu_0[iva], mu_1[iva])
      test = (x[ite], t[ite], y[ite]), (y_cf[ite], mu_0[ite], mu_1[ite])
      yield train, test, valid, self.contfeats, self.binfeats