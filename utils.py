# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist

def minFrobDist_distM(points_A, points_B, metric='euclidean'):

  dist_A = pdist(points_A, metric=metric)
  dist_B = pdist(points_B, metric=metric)

  cost = np.zeros((len(dist_A), len(dist_B)))
  for i in range(len(dist_A)):
    for j in range(len(dist_B)):
      cost[i,j] = (dist_A[i]- dist_B[j])**2

  row_ind, col_ind = linear_sum_assignment(cost)
  minimum_dist = np.sqrt(cost[row_ind, col_ind].sum())
  
  return minimum_dist

