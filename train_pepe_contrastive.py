#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random

import hdbscan
import kornia
import numpy as np
import torch
import torch.nn.functional as F_func
from random import randint
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from tqdm import tqdm

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import render_contrastive_feature
from scene import Scene, GaussianModel, FeatureGaussianModel
from utils.general_utils import safe_state

# Borrowed from GARField but modified
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_quantile_func(scales: torch.Tensor, distribution="normal"):
    """
    Use 3D scale statistics to normalize scales -- use quantile transformer.
    """
    scales = scales.flatten()

    scales = scales.detach().cpu().numpy()

    # Calculate quantile transformer
    quantile_transformer = QuantileTransformer(output_distribution=distribution)
    quantile_transformer = quantile_transformer.fit(scales.reshape(-1, 1))

    def quantile_transformer_func(scales):
        # This function acts as a wrapper for QuantileTransformer.
        # QuantileTransformer expects a numpy array, while we have a torch tensor.
        scales = scales.reshape(-1,1)
        return torch.Tensor(
            quantile_transformer.transform(scales.detach().cpu().numpy())
        ).to(scales.device)

    return quantile_transformer_func

# Prototype bank and clustering helpers for the view-consistent centroid loss (Sec. 4.2.3)
class PrototypeBank:
    def __init__(self, dim, tau_merge=0.9, ema=0.9, max_buf=20000):
        self.dim = dim
        self.tau_merge = tau_merge
        self.ema = ema
        self.max_buf = max_buf
        self.buf = []
        self.centroids = None

    @torch.no_grad()
    def add_candidates(self, tensors_list):
        for t in tensors_list:
            if t is None or t.numel() == 0:
                continue
            t = torch.nn.functional.normalize(t, dim=-1)
            self.buf.append(t.detach())
        total = sum(x.shape[0] for x in self.buf)
        if total > self.max_buf:
            self.buf = self.buf[-1:]

    @torch.no_grad()
    def _greedy_merge(self, X):
        if X.numel() == 0: return None
        idx = torch.randperm(X.shape[0], device=X.device)
        C = []
        for i in idx:
            x = X[i:i+1]
            if len(C) == 0:
                C.append(x)
                continue
            M = torch.cat(C, dim=0)
            sim = (M @ x.T).squeeze(1)
            j = torch.argmax(sim)
            if sim[j] >= self.tau_merge:
                print(f"[greedy_merge] Similar feature : sim={sim[j].item():.4f} (tau_merge={self.tau_merge}) -> Merged with centroid {j}")
                C[j] = self.ema * C[j] + (1 - self.ema) * x
                C[j] = torch.nn.functional.normalize(C[j], dim=-1, eps=1e-6)
            else:
                C.append(x)
        return torch.cat(C, dim=0)

    @torch.no_grad()
    def finalize(self, device='cuda'):
        if len(self.buf) == 0:
            return
        X = torch.cat(self.buf, dim=0).to(device)
        self.centroids = self._greedy_merge(X)
        self.buf = []
        print("[PrototypeBank] finalized with {} prototypes.".format(self.centroids.shape[0]))

    @torch.no_grad()
    def nearest_proto(self, F):
        C = self.centroids
        if C is None or C.shape[0] == 0:
            return None, None
        S = F @ C.T   # [N,K]
        sim, idx = S.max(dim=1)
        return sim, idx

@torch.no_grad()
def segment_means_pre_gate(F0, mask, min_pixels=10):
    """
    F0: [C,H,W] pre-gate rendered features (float, CUDA)
    mask: [H,W] int (0=bg, >0=seg id)
    returns: means [K,C]
    """
    C, H, W = F0.shape
    F = F0.permute(1, 2, 0).reshape(-1, C)
    ids = mask.reshape(-1).long()
    valid = ids > 0
    if valid.sum() == 0:
        return F.new_zeros((0, C))

    ids_v = ids[valid]
    F_v = F[valid]

    Kmax = int(ids_v.max().item()) + 1
    sum_feat = F.new_zeros((Kmax, C))
    cnt = ids_v.new_zeros((Kmax,), dtype=torch.int32)

    sum_feat.index_add_(0, ids_v, F_v)
    cnt.index_add_(0, ids_v, torch.ones_like(ids_v, dtype=cnt.dtype))

    keep = cnt >= min_pixels
    if keep.any():
        means = sum_feat[keep] / cnt[keep].unsqueeze(1).float()
        means = torch.nn.functional.normalize(means, dim=-1)
        return means
    else:
        return F.new_zeros((0, C))

@torch.no_grad()
def segment_means_hdbscan(F0, mask=None, min_cluster_size=10, min_samples=5, 
                          pca_dim=16, use_pca=True, max_samples=16384):
    """
    Cluster the gated feature map with HDBSCAN and return the per-cluster mean features,
    i.e. the candidate centroids of Sec. 4.2.3. Pixels are randomly subsampled at the original
    resolution so the features themselves are never interpolated.

    F0:               [C, H, W] gated rendered features (float, CUDA)
    mask:             [H, W] optional validity map; only pixels with mask > 0 are clustered
    min_cluster_size: HDBSCAN minimum cluster size
    min_samples:      HDBSCAN min_samples
    pca_dim/use_pca:  optionally cluster in a PCA-reduced space (means stay in feature space)
    max_samples:      cap on the number of pixels handed to HDBSCAN (runtime control)

    returns:          means [K, C], L2-normalised
    """
    C, H, W = F0.shape
    flat = F0.reshape(C, -1)

    if mask is not None:
        valid_idx = torch.nonzero(mask.reshape(-1) > 0, as_tuple=False).squeeze(1)
    else:
        valid_idx = torch.arange(flat.shape[1], device=F0.device)

    N = int(valid_idx.numel())
    if N < min_cluster_size:
        return F0.new_zeros((0, C))

    if N > max_samples:
        valid_idx = valid_idx[torch.randperm(N, device=valid_idx.device)[:max_samples]]
        N = max_samples

    F = flat[:, valid_idx].T.contiguous()
    F_np = F.cpu().numpy()
    F_subsample = F_np
    F_pca = None
    if use_pca and C > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=42)
        F_pca = pca.fit_transform(F_subsample)
    F_for_clustering = F_pca if F_pca is not None else F_subsample
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        core_dist_n_jobs=1
    )
    labels = clusterer.fit_predict(F_for_clustering)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]    
    if len(unique_labels) == 0:
        return F.new_zeros((0, C))
    
    cluster_means = []
    for label in unique_labels:
        mask_label = (labels == label)
        # mean in feature space, even when clustering used PCA
        cluster_feats = F_subsample[mask_label]
        mean_feat = cluster_feats.mean(axis=0)
        cluster_means.append(mean_feat)
    
    means = torch.from_numpy(np.stack(cluster_means, axis=0)).to(F.device, dtype=F.dtype)
    means = torch.nn.functional.normalize(means, dim=-1)
    
    return means

def _channel_align(x, target_c):
    return x[..., :target_c]

def training(dataset, opt, pipe, load_iteration, saving_iterations, checkpoint_iterations, debug_from,
             opt_ablate_perception=False, opt_ablate_consistency=False):
    
    proto_bank_upper = PrototypeBank(dim=dataset.feature_dim, tau_merge=0.90, ema=0.9, max_buf=30000)
    proto_bank_lower = PrototypeBank(dim=dataset.feature_dim, tau_merge=0.95, ema=0.9, max_buf=30000)
    G_STEP = 200
    WARMUP = 7000
    Ns_sparse = 50000
    tau_g = 0.90
    LAMBDA_PROXY_UPPER = 0.3
    LAMBDA_PROXY_LOWER = 0.1
    LAMBDA_DEPTH = 0.2
    LAMBDA_L3D = 1.0

    if opt_ablate_perception:
        LAMBDA_DEPTH = 0.0
    if opt_ablate_consistency:
        LAMBDA_PROXY_UPPER = 0.0
        LAMBDA_PROXY_LOWER = 0.0
    
    assert opt.ray_sample_rate > 0 or opt.num_sampled_rays > 0

    dataset.need_masks = True
    dataset.need_masks_scale = True
    dataset.need_depth = True
    
    boundary_sampling_interval = 20

    gaussians = GaussianModel(dataset.sh_degree)
    feature_gaussians = FeatureGaussianModel(dataset.feature_dim)

    sample_rate = 0.2 if 'Replica' in dataset.source_path else 1.0 # skip some images for large-scale dataset
    scene = Scene(dataset, gaussians, feature_gaussians, load_iteration=load_iteration, shuffle=False, target='contrastive_feature', mode='train', sample_rate=sample_rate)

    feature_gaussians.change_to_segmentation_mode(opt, "contrastive_feature", fixed_feature=False)

    scale_gate = torch.nn.Sequential(
        torch.nn.Linear(1, 32, bias=True),
        torch.nn.Sigmoid()
    )
    scale_gate = scale_gate.cuda()
    scale_gate.train()

    param_group = {'params': scale_gate.parameters(), 'lr': opt.feature_lr, 'name': 'f'}
    feature_gaussians.optimizer.add_param_group(param_group)

    del gaussians
    torch.cuda.empty_cache()

    background = torch.ones([dataset.feature_dim], dtype=torch.float32, device="cuda") if dataset.white_background else torch.zeros([dataset.feature_dim], dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    first_iter = 0
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_stack = None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    all_scales = []
    for cam in scene.getTrainCameras():
        all_scales.append(cam.mask_scales)
    all_scales = torch.cat(all_scales)
    upper_bound_scale = all_scales.max().item()
    scale_aware_dim = opt.scale_aware_dim

    if scale_aware_dim <= 0 or scale_aware_dim >= 32:
        q_trans = get_quantile_func(all_scales, "uniform")
    else:
        q_trans = get_quantile_func(all_scales, "uniform")
        fixed_scale_gate = torch.tensor([[1 for j in range(32 - scale_aware_dim + i)] + [0 for k in range(scale_aware_dim - i)] for i in range(scale_aware_dim+1)]).cuda()

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        
        rand_idx = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        
        with torch.no_grad():
            sam_masks = viewpoint_cam.original_masks.cuda().float()
            viewpoint_cam.feature_height, viewpoint_cam.feature_width = viewpoint_cam.image_height, viewpoint_cam.image_width
            mask_scales = viewpoint_cam.mask_scales.cuda()
            mask_scales, sort_indices = torch.sort(mask_scales, descending=True)
            sam_masks = sam_masks[sort_indices, :, :]
            num_sampled_scales = 8
            sampled_scale_index = torch.randperm(len(mask_scales))[:num_sampled_scales]

            tmp = torch.zeros(num_sampled_scales+2)
            tmp[1:len(sampled_scale_index)+1] = sampled_scale_index
            tmp[-1] = len(mask_scales) - 1
            tmp[0] = -1
            sampled_scale_index = tmp.long()
            sampled_scales = mask_scales[sampled_scale_index]
            second_big_scale = mask_scales[mask_scales < upper_bound_scale].max()

            if iteration % boundary_sampling_interval == 0:
                ray_sample_rate = opt.ray_sample_rate if opt.ray_sample_rate > 0 else opt.num_sampled_rays / (2*(sam_masks.shape[-1] * sam_masks.shape[-2]))
            else:
                ray_sample_rate = opt.ray_sample_rate if opt.ray_sample_rate > 0 else opt.num_sampled_rays / (sam_masks.shape[-1] * sam_masks.shape[-2])

            sampled_ray = torch.rand(sam_masks.shape[-2], sam_masks.shape[-1]).cuda() < ray_sample_rate
            non_mask_region = sam_masks.sum(dim=0) == 0
            sampled_ray_with_background = sampled_ray.clone()
            sampled_ray = torch.logical_and(sampled_ray, ~non_mask_region)

            per_pixel_mask_size = sam_masks * sam_masks.sum(-1).sum(-1)[:,None,None]
            per_pixel_mean_mask_size = per_pixel_mask_size.sum(dim = 0) / (sam_masks.sum(dim = 0) + 1e-9)
            per_pixel_mean_mask_size = per_pixel_mean_mask_size[sampled_ray]
            pixel_to_pixel_mask_size = per_pixel_mean_mask_size.unsqueeze(0) * per_pixel_mean_mask_size.unsqueeze(1)
            ptp_max_size = pixel_to_pixel_mask_size.max()
            pixel_to_pixel_mask_size[pixel_to_pixel_mask_size == 0] = 1e10
            per_pixel_weight = torch.clamp(ptp_max_size / pixel_to_pixel_mask_size, 1.0, None)
            per_pixel_weight = (per_pixel_weight - per_pixel_weight.min()) / (per_pixel_weight.max() - per_pixel_weight.min()) * 9. + 1.
            sam_masks_sampled_ray = sam_masks[:, sampled_ray]

            gt_corrs = []
            sampled_scales[0] = upper_bound_scale + upper_bound_scale * torch.rand(1)[0]
            for idx, si in enumerate(sampled_scale_index):
                upper_bound = sampled_scales[idx] >= upper_bound_scale

                if si != len(mask_scales) - 1 and not upper_bound:
                    sampled_scales[idx] -= (sampled_scales[idx] - mask_scales[si+1]) * torch.rand(1)[0]
                elif upper_bound:
                    sampled_scales[idx] -= (sampled_scales[idx] - second_big_scale) * torch.rand(1)[0]
                else:
                    sampled_scales[idx] -= sampled_scales[idx] * torch.rand(1)[0]

                if not upper_bound:
                    gt_vec = torch.zeros_like(sam_masks_sampled_ray)
                    gt_vec[:si+1,:] = sam_masks_sampled_ray[:si+1,:]
                    for j in range(si, -1, -1):
                        gt_vec[j,:] = torch.logical_and(
                            torch.logical_not(gt_vec[j+1:,:].any(dim = 0)), gt_vec[j,:]
                        )
                    gt_vec[si+1:,:] = sam_masks_sampled_ray[si+1:,:]
                else:
                    gt_vec = sam_masks_sampled_ray

                gt_corr = torch.einsum('nh,nj->hj', gt_vec, gt_vec)
                gt_corr[gt_corr != 0] = 1
                gt_corrs.append(gt_corr)

            gt_corrs = torch.stack(gt_corrs, dim = 0)

            # physical scales before the quantile transform: needed to decide which masks are
            # "at least as coarse as the query scale" when building the Λ(s, p) id map below.
            sampled_scales_phys = sampled_scales.clone()

            sampled_scales = q_trans(sampled_scales).squeeze()
            sampled_scales = sampled_scales.squeeze()

            gt_image = viewpoint_cam.original_image.cuda()
            rel_depth = viewpoint_cam.depth.cuda()
            mask_H, mask_W = sampled_ray_with_background.shape
            depth_H, depth_W = rel_depth.shape[1], rel_depth.shape[2]
            
            if (mask_H != depth_H) or (mask_W != depth_W):
                rel_depth = F_func.interpolate(
                    rel_depth.unsqueeze(0),
                    size=(mask_H, mask_W),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                
                gt_image = F_func.interpolate(
                    gt_image.unsqueeze(0),
                    size=(mask_H, mask_W),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            
            depth_4d = rel_depth.unsqueeze(0)
            
            _, canny_edges = kornia.filters.canny(depth_4d)
            boundary_map = canny_edges.squeeze(0).squeeze(0).bool()
            boundary_map[:2, :] = 0
            boundary_map[-2:, :] = 0
            boundary_map[:, :2] = 0
            boundary_map[:, -2:] = 0
            
            kernel_size = 5
            dilation_kernel = torch.ones(1, 1, kernel_size, kernel_size, device='cuda')
            pad_size = kernel_size // 2
            boundary_padded = F_func.pad(boundary_map.float().unsqueeze(0).unsqueeze(0), 
                                  pad=(pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
            boundary_thick = F_func.conv2d(boundary_padded, dilation_kernel, padding=0)
            boundary_thick = (boundary_thick >= 1).squeeze().bool()
            
            interior_map = ~boundary_thick
            interior_map = interior_map[sampled_ray_with_background]
            interior_map = interior_map.unsqueeze(1) * interior_map.unsqueeze(0)

            # Scale-aware local adaptive depth + color similarity
            H, W = rel_depth.shape[1], rel_depth.shape[2]
            sampled_coords = torch.nonzero(sampled_ray_with_background, as_tuple=False)
            sampled_depth_values = rel_depth[:, sampled_ray_with_background].squeeze(0)
            sampled_rgb = gt_image[:, sampled_ray_with_background]
            sampled_rgb = sampled_rgb.permute(1, 0)
            sampled_color = sampled_rgb
            coords_i = sampled_coords.unsqueeze(1).float()
            coords_j = sampled_coords.unsqueeze(0).float()
            spatial_dist = torch.norm(coords_i - coords_j, dim=2, p=2)
            depth_i = sampled_depth_values.unsqueeze(1)
            depth_j = sampled_depth_values.unsqueeze(0)
            depth_diff = torch.abs(depth_i - depth_j)
            color_i = sampled_color.unsqueeze(1)
            color_j = sampled_color.unsqueeze(0)
            color_diff = torch.norm(color_i - color_j, dim=2, p=2)
            
            num_scales = len(sampled_scales)
            scale_ratios = sampled_scales
            
            min_window_ratio = 0.03
            max_window_ratio = 0.15
            window_ratios = min_window_ratio + scale_ratios * (max_window_ratio - min_window_ratio)
            local_window_sizes = (min(H, W) * window_ratios).int()
            local_window_sizes = torch.clamp(local_window_sizes, 20, 200)
            local_masks = spatial_dist.unsqueeze(0) <= local_window_sizes.view(-1, 1, 1)
            depth_diff_expanded = depth_diff.unsqueeze(0)

            local_depth_diff = depth_diff_expanded * local_masks.float()
            local_count = local_masks.sum(dim=2, keepdim=True).clamp(min=1)
            local_mean_depth_diff = local_depth_diff.sum(dim=2, keepdim=True) / local_count
            local_depth_diff_sq = (depth_diff_expanded ** 2) * local_masks.float()
            local_mean_depth_diff_sq = local_depth_diff_sq.sum(dim=2, keepdim=True) / local_count
            local_std_depth_diff = torch.sqrt(torch.clamp(local_mean_depth_diff_sq - local_mean_depth_diff ** 2, min=0))

            k_depth = 1.0  # Sensitivity parameter
            adaptive_threshold_i = local_mean_depth_diff + k_depth * local_std_depth_diff
            adaptive_threshold_j = adaptive_threshold_i.transpose(1, 2)
            adaptive_depth_threshold = torch.max(adaptive_threshold_i, adaptive_threshold_j)

            color_diff_expanded = color_diff.unsqueeze(0)  # [1, N, N]

            local_color_diff = color_diff_expanded * local_masks.float()
            local_mean_color_diff = local_color_diff.sum(dim=2, keepdim=True) / local_count
            local_color_diff_sq = (color_diff_expanded ** 2) * local_masks.float()
            local_mean_color_diff_sq = local_color_diff_sq.sum(dim=2, keepdim=True) / local_count
            local_std_color_diff = torch.sqrt(torch.clamp(local_mean_color_diff_sq - local_mean_color_diff ** 2, min=0))

            k_color = 1.0  # Sensitivity parameter
            adaptive_color_threshold_i = local_mean_color_diff + k_color * local_std_color_diff
            adaptive_color_threshold_j = adaptive_color_threshold_i.transpose(1, 2)
            adaptive_color_threshold = torch.max(adaptive_color_threshold_i, adaptive_color_threshold_j)

            interior_map_expanded = interior_map.unsqueeze(0)  # [1, N, N]
            depth_similar = (depth_diff_expanded < adaptive_depth_threshold) & local_masks & interior_map_expanded
            color_similar = (color_diff_expanded < adaptive_color_threshold) & local_masks & interior_map_expanded
            depth_color_similar_all_scales = depth_similar & color_similar
            depth_dissimilar = (depth_diff_expanded >= adaptive_depth_threshold) & local_masks & interior_map_expanded
            color_dissimilar = (color_diff_expanded >= adaptive_color_threshold) & local_masks & interior_map_expanded
            depth_color_dissimilar_all_scales = depth_dissimilar & color_dissimilar
            depth_color_similar_all_scales = torch.triu(depth_color_similar_all_scales, diagonal=1)
            depth_color_dissimilar_all_scales = torch.triu(depth_color_dissimilar_all_scales, diagonal=1)

        render_pkg_feat = render_contrastive_feature(viewpoint_cam, feature_gaussians, pipe, background, norm_point_features=True, smooth_weights=None, smooth_type='traditional', smooth_K=opt.smooth_K)
        rendered_features = render_pkg_feat["render"]
        visibility_filter = render_pkg_feat["visibility_filter"]
        radii = render_pkg_feat["radii"]

        rendered_feature_norm = rendered_features.norm(dim = 0, p=2).mean()
        rendered_feature_norm_reg = (1-rendered_feature_norm)**2

        # 3D Gaussian feature norm regularization (prevent dominance in rendering)
        # feature_gaussians._features shape: [N_gaussians, C]
        gaussian_3d_features = feature_gaussians.get_point_features  # [N, C]
        gaussian_3d_feature_norms = torch.norm(gaussian_3d_features, dim=1, p=2)  # [N]
        # Encourage each Gaussian's feature to have norm close to 1
        loss_3d_feature_norm = ((gaussian_3d_feature_norms - 1.0) ** 2).mean()

        rendered_features = torch.nn.functional.interpolate(rendered_features.unsqueeze(0), viewpoint_cam.original_masks.shape[-2:], mode='bilinear').squeeze(0)

        gates = scale_gate(sampled_scales.unsqueeze(-1))
        feat_wscale = rendered_features.unsqueeze(0).repeat([sampled_scales.shape[0],1,1,1])
        feat_wscale = feat_wscale * gates.unsqueeze(-1).unsqueeze(-1)
        sampled_feat_wscale = feat_wscale[:,:,sampled_ray]
        sampled_feat_wscale = sampled_feat_wscale.permute([0,2,1])
        norm_sampled_feat_wscale = torch.nn.functional.normalize(sampled_feat_wscale, dim=-1, p=2)
        corr = torch.einsum('nhc,njc->nhj', norm_sampled_feat_wscale, norm_sampled_feat_wscale)

        diag_mask = torch.eye(corr.shape[1], dtype=bool, device=corr.device)

        sum_0 = gt_corrs.sum(dim = 0)
        consistent_negative = sum_0 == 0
        consistent_positive = sum_0 == len(gt_corrs)
        inconsistent = torch.logical_not(torch.logical_or(consistent_negative, consistent_positive))
        inconsistent_num = inconsistent.count_nonzero()
        sampled_num = inconsistent_num / 2

        rand_num = torch.rand_like(sum_0)

        sampled_positive = torch.logical_and(consistent_positive, rand_num < sampled_num / consistent_positive.count_nonzero())

        sampled_negative = torch.logical_and(consistent_negative, rand_num < sampled_num / consistent_negative.count_nonzero())

        sampled_mask_positive = torch.logical_or(
            torch.logical_or(
                sampled_positive, torch.any(torch.logical_and(corr < 0.75, gt_corrs == 1), dim = 0)
            ), 
            inconsistent
        )
        sampled_mask_positive = torch.logical_and(sampled_mask_positive, ~diag_mask)
        sampled_mask_positive = torch.triu(sampled_mask_positive, diagonal=0)
        sampled_mask_positive = sampled_mask_positive.bool()

        sampled_mask_negative = torch.logical_or(
            torch.logical_or(
                sampled_negative, torch.any(torch.logical_and(corr > 0.5, gt_corrs == 0), dim = 0)
            ), 
            inconsistent
        )
        sampled_mask_negative = torch.logical_and(sampled_mask_negative, ~diag_mask)
        sampled_mask_negative = torch.triu(sampled_mask_negative, diagonal=0)
        sampled_mask_negative = sampled_mask_negative.bool()
        
        per_pixel_weight = per_pixel_weight.unsqueeze(0)
        loss_pos_base = (-1.0 * per_pixel_weight[:, sampled_mask_positive] * gt_corrs[:, sampled_mask_positive] * corr[:, sampled_mask_positive]).mean()
        loss_neg_base = (1.0 * per_pixel_weight[:, sampled_mask_negative] * (1 - gt_corrs[:, sampled_mask_negative]) * torch.relu(corr[:, sampled_mask_negative])).mean()

        # Scale-aware depth similarity loss
        sampled_feat_with_bg = rendered_features[:, sampled_ray_with_background]
        sampled_feat_with_bg = sampled_feat_with_bg.permute(1, 0)
        
        with torch.no_grad():
            mask_relationships = []
            for idx in range(len(sampled_scales)):
                # masks are scale-descending, so those with s_Mk >= s are exactly indices 0
                cut = int((mask_scales >= sampled_scales_phys[idx]).sum().item()) - 1
                cut = max(cut, 0)
                covering = sam_masks[:cut + 1][:, sampled_ray_with_background]
                rank = torch.arange(1, cut + 2, device=covering.device,
                                    dtype=covering.dtype).unsqueeze(1)
                sampled_mask_ids = (covering * rank).amax(dim=0)

                mask_i = sampled_mask_ids.unsqueeze(1)
                mask_j = sampled_mask_ids.unsqueeze(0)
                
                same_mask = (mask_i == mask_j) & (mask_i > 0) & (mask_j > 0)
                different_mask = (mask_i != mask_j) & (mask_i > 0) & (mask_j > 0)                
                background_involved = (mask_i == 0) | (mask_j == 0)
                
                mask_relationships.append({
                    'same_mask': same_mask,
                    'different_mask': different_mask,
                    'background_involved': background_involved
                })
        
        loss_depth = torch.tensor(0.0, device=sampled_feat_with_bg.device)
        
        for scale_idx in range(len(sampled_scales)):
            g = gates[scale_idx]
            feat_scaled = sampled_feat_with_bg * g
            norm_feat_scaled = torch.nn.functional.normalize(feat_scaled, dim=-1, p=2)
            
            # Compute feature correlation for this scale
            corr_depth = torch.mm(norm_feat_scaled, norm_feat_scaled.T)
            depth_color_similar_scale = depth_color_similar_all_scales[scale_idx]
            depth_color_dissimilar_scale = depth_color_dissimilar_all_scales[scale_idx]            
            same_mask = mask_relationships[scale_idx]['same_mask']
            different_mask = mask_relationships[scale_idx]['different_mask']
            background_involved = mask_relationships[scale_idx]['background_involved']            
            depth_similar_valid = depth_color_similar_scale & (same_mask | background_involved)
            depth_similar_count = depth_similar_valid.sum()
            
            if depth_similar_count > 0:
                loss_depth += 1.0 * (1.0 - corr_depth[depth_similar_valid]).mean()
            
            depth_dissimilar_valid = depth_color_dissimilar_scale & (different_mask | background_involved)
            depth_dissimilar_count = depth_dissimilar_valid.sum()
            
            if depth_dissimilar_count > 0:
                loss_depth += 1.0 * torch.relu(corr_depth[depth_dissimilar_valid]).mean()
        
        loss_depth = loss_depth / len(sampled_scales)        
        loss_feature = loss_pos_base + loss_neg_base + opt.rfn * rendered_feature_norm_reg

        with torch.no_grad():
            cosine_pos = corr[gt_corrs == 1].mean()
            cosine_neg = corr[gt_corrs == 0].mean()

        L_gfl_proxy_upper = torch.tensor(0.0, device='cuda')
        L_gfl_proxy_lower = torch.tensor(0.0, device='cuda')
        
        if iteration >= WARMUP:
            upper_scale_idx = 0
            g_upper = gates[upper_scale_idx]
            F0_upper = rendered_features * g_upper.unsqueeze(-1).unsqueeze(-1)
            F0_upper = F0_upper[:, sampled_ray]
            F0_upper = F0_upper.permute(1, 0)
            F0_upper = torch.nn.functional.normalize(F0_upper, dim=-1, p=2)
            
            lower_scale_idx = len(sampled_scales) - 1
            g_lower = gates[lower_scale_idx]
            F0_lower = rendered_features * g_lower.unsqueeze(-1).unsqueeze(-1)
            F0_lower = F0_lower[:, sampled_ray]
            F0_lower = F0_lower.permute(1, 0)
            F0_lower = torch.nn.functional.normalize(F0_lower, dim=-1, p=2)
            
            # Upper scale GFL loss
            if proto_bank_upper.centroids is not None and proto_bank_upper.centroids.numel() > 0:
                sim_p, idx_p = proto_bank_upper.nearest_proto(F0_upper)
                if sim_p is not None:
                    pos_mask = sim_p > tau_g
                    if pos_mask.any():
                        L_gfl_proxy_upper = L_gfl_proxy_upper + (1.0 - sim_p[pos_mask]).mean()

            # Lower scale GFL loss
            if proto_bank_lower.centroids is not None and proto_bank_lower.centroids.numel() > 0:
                sim_p, idx_p = proto_bank_lower.nearest_proto(F0_lower)
                if sim_p is not None:
                    pos_mask = sim_p > tau_g
                    if pos_mask.any():
                        L_gfl_proxy_lower = L_gfl_proxy_lower + (1.0 - sim_p[pos_mask]).mean()


        if (iteration % G_STEP == 0) and (iteration >= WARMUP):
            with torch.no_grad():
                upper_scale_idx = 0
                lower_scale_idx = len(sampled_scales) - 1
                
                valid_region = sam_masks.any(dim=0).long()
                dense_map_upper = valid_region
                
                g_upper = gates[upper_scale_idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                F0_upper_full = rendered_features.unsqueeze(0) * g_upper
                F0_upper_full = F0_upper_full.squeeze(0)
                F0_upper_full = torch.nn.functional.normalize(F0_upper_full, dim=0, p=2)
                
                means_upper = segment_means_hdbscan(
                    F0_upper_full, 
                    mask=dense_map_upper,
                    min_cluster_size=20,
                    pca_dim=16,
                    use_pca=False
                )
                if means_upper is not None and means_upper.numel() > 0:
                    proto_bank_upper.add_candidates([means_upper])
                proto_bank_upper.finalize(device='cuda')
                
                # Same spatial region as the upper bank; the two differ only by the scale gate ψ(s).
                dense_map_lower = valid_region

                g_lower = gates[lower_scale_idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1,32,1,1]
                F0_lower_full = rendered_features.unsqueeze(0) * g_lower  # [1,C,H,W]
                F0_lower_full = F0_lower_full.squeeze(0)  # [C,H,W]
                F0_lower_full = torch.nn.functional.normalize(F0_lower_full, dim=0, p=2)
                
                means_lower = segment_means_hdbscan(
                    F0_lower_full, 
                    mask=dense_map_lower,
                    min_cluster_size=10,
                    pca_dim=32,
                    use_pca=False
                )
                if means_lower is not None and means_lower.numel() > 0:
                    proto_bank_lower.add_candidates([means_lower])
                proto_bank_lower.finalize(device='cuda')

        total_loss = loss_feature \
                   + LAMBDA_PROXY_UPPER * L_gfl_proxy_upper \
                   + LAMBDA_PROXY_LOWER * L_gfl_proxy_lower \
                   + LAMBDA_DEPTH * loss_depth \
                   + LAMBDA_L3D * loss_3d_feature_norm

        total_loss.backward()

        feature_gaussians.optimizer.step()
        feature_gaussians.optimizer.zero_grad(set_to_none = True)

        iter_end.record()
        
        if iteration % 10 == 0:
            progress_bar.set_postfix({
                "RFN": f"{rendered_feature_norm.item():.{3}f}",
                "Pos cos": f"{cosine_pos.item():.{3}f}",
                "Neg cos": f"{cosine_neg.item():.{3}f}",
                "Loss": f"{total_loss.item():.{3}f}",
            })
            progress_bar.update(10)

    scene.save_feature(iteration, target='contrastive_feature', smooth_type='traditional', smooth_K=opt.smooth_K)
    save_dir = os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration))
    os.makedirs(save_dir, exist_ok=True)
    torch.save(scale_gate.state_dict(), os.path.join(save_dir, "scale_gate.pt"))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=np.random.randint(10000, 20000))
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--target', default='contrastive_feature', const='contrastive_feature', nargs='?', choices=['contrastive_feature'])
    parser.add_argument("--load_iteration", default=-1, type=int)
    parser.add_argument("--seed", default=0, type=int,
                        help="Random seed. Default 0 matches safe_state(), i.e. the published runs.")
    parser.add_argument("--ablate_perception_loss", action="store_true",
                        help="Table 7 ablation: drop the perception contrastive loss L_p (Sec. 4.2.2).")
    parser.add_argument("--ablate_consistency_loss", action="store_true",
                        help="Table 7 ablation: drop the view-consistent centroid loss L_c (Sec. 4.2.3).")
    
    args = get_combined_args(parser, target_cfg_file = 'cfg_args')
    
    # Set seed using the argument
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    set_seed(args.seed)
    
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.load_iteration,
             args.save_iterations, args.checkpoint_iterations, args.debug_from,
             opt_ablate_perception=args.ablate_perception_loss,
             opt_ablate_consistency=args.ablate_consistency_loss)

    # All done
    print("\nTraining complete.")
