% The Symmetric Grouped Mixtures Clustering (S-GMC) clustering algorithm.  
%   This function uses the Symmetric Grouped Mixtures Clustering model [1] to
%   cluster multiple datasets simultaneously with cluster sharing between
%   datasets. It uses a symmetric Dirichlet prior over the group mixture
%   weights, and a Gaussian-Wishart prior over the cluster parameters. This
%   algorithm is similar to latent Dirichlet allocation with Gaussian
%   observations. 
%
%   This is referred to Gaussian Latent Dirichlet Allocation (G-LDA) in [2, 3].
%
%   [qZ, weights, means, covariances] = learnSGMC (X, options)
%
% Arguments:
%  - X, {Jx[NxD]} cell array of observation matrices (one cell for each group)
%  - options, structure with members (all are optional):
%     + prior, [double] prior cluster value (1 default)
%     + verbose, [bool] verbose output flag (false default)
%     + sparse, [bool] do fast but approximate sparse VB updates (false default)
%     + threads, [unsigned int] number of threads to use (automatic default)
%
% Returns
%  - qZ, {Jx[NxK]} cell array of assignment probablities
%  - weights, {Jx[1xK]} Group mixture weights
%  - means, {Kx[1xD]} Gaussian mixture means
%  - covariances, {Kx[DxD]} Gaussian mixture covariances
%
% Author: Daniel Steinberg
%         Australian Centre for Field Robotics
%         University of Sydney
%
% Date:   27/09/2012
%
% References:
%  [1] D. M. Steinberg, An Unsupervised Approach to Modelling Visual Data, PhD
%      Thesis, 2013.
%  [2] Synergistic Clustering of Image and Segment Descriptors for Unsupervised
%      Scene Understanding. D. M. Steinberg, O. Pizarro, S. B. Williams. In 
%      International Conference on Computer Vision (ICCV). IEEE, Sydney, NSW, 
%      2013.
%  [3] D. M. Steinberg, O. Pizarro, S. B. Williams. Hierarchical Bayesian 
%      Models for Unsupervised Scene Understanding. Journal of Computer Vision
%      and Image Understanding (CVIU). Elsevier, 2014.

% libcluster -- A collection of hierarchical Bayesian clustering algorithms.
% Copyright (C) 2013 Daniel M. Steinberg (daniel.m.steinberg@gmail.com)
%
% This file is part of libcluster.
%
% libcluster is free software: you can redistribute it and/or modify it under
% the terms of the GNU Lesser General Public License as published by the Free
% Software Foundation, either version 3 of the License, or (at your option)
% any later version.
%
% libcluster is distributed in the hope that it will be useful, but WITHOUT
% ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
% FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
% for more details.
%
% You should have received a copy of the GNU Lesser General Public License
% along with libcluster. If not, see <http://www.gnu.org/licenses/>.
