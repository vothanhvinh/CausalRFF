# -*- coding: utf-8 -*-

import numpy as np

class Evaluation():
  def __init__(self, m0, m1, y0, y1):
    self.m0 = m0
    self.m1 = m1
    self.y0 = y0
    self.y1 = y1

  def pehe(self, y0pred, y1pred):
    return np.sqrt(np.mean(((y1pred - y0pred) - (self.m1 - self.m0))**2))

  def absolute_err_ate(self, y0pred, y1pred):
    return np.abs(np.mean(y1pred - y0pred) - np.mean(self.m1 - self.m0))