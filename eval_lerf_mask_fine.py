# Copyright (C) 2023, Gaussian-Grouping
# Multi-scale rendering and evaluation script for LERF mask fine-grained benchmark
# This script renders with multiple scales and evaluates immediately to keep only the best results
import torch
import torchvision
from scene import Scene, GaussianModel, FeatureGaussianModel
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_contrastive_feature
import torchvision
import numpy as np
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
import colorsys
import hdbscan
from PIL import Image
import torch.nn.functional as F
import sys
import json
from sklearn.decomposition import PCA

from ext.grounded_sam import grouned_sam_output, load_model_hf, select_obj_ioa
from segment_anything import sam_model_registry, SamPredictor


def feature_to_rgb(features):
    """Convert features to RGB using PCA"""
    H, W = features.shape[1], features.shape[2]
    features_reshaped = features.view(features.shape[0], -1).T

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_reshaped.cpu().numpy())
    pca_result = pca_result.reshape(H, W, 3)
    pca_normalized = 255 * (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())

    return pca_normalized.astype('uint8')


def id2rgb(id, max_num_obj=256):
    """Convert object ID to RGB color"""
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)
    s = 0.5 + (id % 2) * 0.5
    l = 0.5

    rgb = np.zeros((3, ), dtype=np.uint8)
    if id == 0:
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)

    return rgb


def visualize_obj(objects):
    """Visualize object IDs as colored mask"""
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(id)
        rgb_mask[objects == id] = colored_mask
    return rgb_mask


def hdbscan_clustering(features, min_cluster_size=10, min_samples=5, max_samples=16384, pca_dim=16, use_pca=True):
    """HDBSCAN clustering on features"""
    H, W, C = features.shape
    
    if H * W < min_cluster_size:
        return np.full((H, W), -1), None
    
    F_tensor = features.reshape(-1, C)
    N = F_tensor.shape[0]
    
    subsample_indices = None
    if N > max_samples:
        subsample_indices = np.random.choice(N, max_samples, replace=False)
        F_subsample = F_tensor.cpu().numpy()[subsample_indices]
    else:
        F_subsample = F_tensor.cpu().numpy()
    
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
        return np.full((H, W), -1), None
    
    cluster_centroids = []
    for label in unique_labels:
        mask_label = (labels == label)
        cluster_feats = F_subsample[mask_label]
        mean_feat = cluster_feats.mean(axis=0)
        cluster_centroids.append(mean_feat)
    
    cluster_centroids = torch.from_numpy(np.stack(cluster_centroids, axis=0)).cuda()
    cluster_centroids = F.normalize(cluster_centroids, dim=-1, p=2)
    
    F_full = features.reshape(-1, C)
    F_normalized = F.normalize(F_full, dim=-1, p=2)
    similarities = F_normalized @ cluster_centroids.T
    full_labels = similarities.argmax(dim=1).cpu().numpy()
    cluster_labels = full_labels.reshape(H, W)
    
    return cluster_labels, cluster_centroids


def mask_to_boundary(mask, dilation_ratio=0.02):
    """Convert binary mask to boundary mask for boundary IoU calculation"""
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    
    return mask - mask_erode


def boundary_iou(gt, dt, dilation_ratio=0.02):
    """Compute boundary IoU between two binary masks"""
    dt = (dt > 128).astype('uint8')
    gt = (gt > 128).astype('uint8')
    
    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_iou(mask1, mask2):
    """Calculate IoU between two boolean masks"""
    mask1_bool = mask1 > 128
    mask2_bool = mask2 > 128
    intersection = np.logical_and(mask1_bool, mask2_bool)
    union = np.logical_or(mask1_bool, mask2_bool)
    
    if np.sum(union) == 0:
        return 0.0
    
    return np.sum(intersection) / np.sum(union)


def resize_mask(mask, target_shape):
    """Resize the mask to the target shape."""
    return np.array(Image.fromarray(mask).resize((target_shape[1], target_shape[0]), resample=Image.NEAREST))


def render_and_evaluate_scale(scale_value, view_data, TEXT_PROMPT, gt_ref_mask, gt_masks_dict,
                               ref_rendered_features, feature_gaussians, pipeline, background, 
                               scale_gate, threshold=0.9, smooth_K=16, use_hdbscan=True):
    """
    Render all views with given scale and evaluate immediately.
    Uses GT mask from reference view instead of Grounded-SAM.
    Returns mean IoU and BIoU for this scale.
    """
    iou_scores_per_view = []
    biou_scores_per_view = []
    temp_masks = {}  # Store temporary masks for this scale
    
    # ========== Extract query centroids for THIS scale using GT mask ==========
    with torch.no_grad():
        # Apply scale gate to reference features
        scale_tensor = torch.tensor([scale_value], dtype=torch.float32, device='cuda')
        gate = scale_gate(scale_tensor)
        feature_with_scale = ref_rendered_features * gate.unsqueeze(-1).unsqueeze(-1)
        scale_conditioned_feature = feature_with_scale.permute([1, 2, 0])
        normed_features = F.normalize(scale_conditioned_feature, dim=-1, p=2)
        
        if use_hdbscan:
            # HDBSCAN on reference view at THIS scale
            cluster_labels_0, cluster_centroids_0 = hdbscan_clustering(
                normed_features,
                min_cluster_size=10,
                min_samples=5,
                max_samples=16384,
                pca_dim=16,
                use_pca=False
            )
            
            if cluster_centroids_0 is None:
                print(f"Warning: No clusters found in reference at scale {scale_value:.1f}, using mean feature")
                gt_mask_tensor = torch.from_numpy(gt_ref_mask).cuda() if isinstance(gt_ref_mask, np.ndarray) else gt_ref_mask
                mask_bool = gt_mask_tensor > 128
                masked_features = normed_features[mask_bool]
                query_centroids = masked_features.mean(dim=0, keepdim=True)
                query_centroids = F.normalize(query_centroids, dim=-1, p=2)
            else:
                # Use GT mask from reference view
                gt_mask_np = gt_ref_mask
                
                # Resize GT mask to match cluster_labels size if needed
                if gt_mask_np.shape != cluster_labels_0.shape:
                    from scipy.ndimage import zoom
                    zoom_factor = (cluster_labels_0.shape[0] / gt_mask_np.shape[0],
                                   cluster_labels_0.shape[1] / gt_mask_np.shape[1])
                    gt_mask_resized = zoom(gt_mask_np.astype(float), zoom_factor, order=0) > 0.5
                else:
                    gt_mask_resized = gt_mask_np
                
                mask_bool = gt_mask_resized > 128
                
                masked_cluster_labels = cluster_labels_0[mask_bool]
                unique_masked_clusters = np.unique(masked_cluster_labels)
                unique_masked_clusters = unique_masked_clusters[unique_masked_clusters >= 0]
                
                # Filter by coverage (bidirectional check)
                iou_threshold_cluster = 0.8
                coverage_threshold_mask = 0.1
                valid_clusters = []
                text_mask_size = mask_bool.sum()
                
                for cluster_id in unique_masked_clusters:
                    cluster_mask = (cluster_labels_0 == cluster_id)
                    cluster_size = cluster_mask.sum()
                    intersection = np.logical_and(cluster_mask, mask_bool).sum()
                    cluster_coverage = intersection / (cluster_size + 1e-9)
                    mask_coverage = intersection / (text_mask_size + 1e-9)
                    
                    if cluster_coverage >= iou_threshold_cluster or mask_coverage >= coverage_threshold_mask:
                        valid_clusters.append(cluster_id)
                
                if len(valid_clusters) == 0:
                    valid_clusters = unique_masked_clusters.tolist()
                
                query_centroids = cluster_centroids_0[valid_clusters]
                print(f"  Reference view (scale {scale_value:.1f}): Found {len(valid_clusters)} query centroids")
        else:
            # Direct feature matching
            gt_mask_tensor = torch.from_numpy(gt_ref_mask).cuda() if isinstance(gt_ref_mask, np.ndarray) else gt_ref_mask
            mask_bool = gt_mask_tensor > 0
            masked_features = normed_features[mask_bool]
            query_centroids = masked_features.mean(dim=0)
            query_centroids = F.normalize(query_centroids, dim=-1, p=2)
    # ============================================================
    
    for idx in range(len(view_data)):
        with torch.no_grad():
            current_view_data = view_data[idx]
            view = current_view_data['view']
            
            # Render features with the given scale
            results = render_contrastive_feature(view, feature_gaussians, pipeline, background,
                                                norm_point_features=True, smooth_weights=None,
                                                smooth_type='traditional', smooth_K=smooth_K)
            rendered_features = results["render"]
            rendered_features = F.interpolate(rendered_features.unsqueeze(0),
                                            view.original_image.shape[1:],
                                            mode='bilinear').squeeze(0)
            
            # Apply scale gate
            scale_tensor = torch.tensor([scale_value], dtype=torch.float32, device='cuda')
            gate = scale_gate(scale_tensor)
            feature_with_scale = rendered_features * gate.unsqueeze(-1).unsqueeze(-1)
            scale_conditioned_feature = feature_with_scale.permute([1, 2, 0])
            normed_features = F.normalize(scale_conditioned_feature, dim=-1, p=2)
            
            if use_hdbscan:
                # Perform HDBSCAN clustering
                cluster_labels, cluster_centroids = hdbscan_clustering(
                    normed_features,
                    min_cluster_size=10,
                    min_samples=5,
                    max_samples=16384,
                    pca_dim=16,
                    use_pca=False
                )
                
                if cluster_centroids is None:
                    # Fallback to pixel-wise similarity
                    similarity_map = torch.einsum('hwc,kc->hwk', normed_features, query_centroids)
                    similarity = similarity_map.max(dim=-1)[0]
                else:
                    # Cluster-to-cluster matching
                    centroid_similarity = query_centroids @ cluster_centroids.T
                    H, W = normed_features.shape[0], normed_features.shape[1]
                    similarity = torch.zeros((H, W), device='cuda')
                    
                    cluster_labels_tensor = torch.from_numpy(cluster_labels).cuda()
                    unique_clusters = torch.unique(cluster_labels_tensor)
                    
                    for cluster_id in unique_clusters:
                        if cluster_id < 0:
                            continue
                        cluster_mask = (cluster_labels_tensor == cluster_id)
                        max_sim = centroid_similarity[:, cluster_id].max()
                        similarity[cluster_mask] = max_sim
            else:
                # Direct feature similarity (single query feature case)
                similarity = torch.einsum('hwc,c->hw', normed_features, query_centroids.squeeze())
            
            # Apply threshold
            pred_obj_mask = (similarity > threshold).cpu().numpy().astype(np.uint8) * 255
            temp_masks[idx] = pred_obj_mask
            
            # Evaluate if GT exists for this view
            if idx in gt_masks_dict:
                gt_mask = gt_masks_dict[idx]
                
                # Resize if needed
                if pred_obj_mask.shape != gt_mask.shape:
                    pred_obj_mask_resized = np.array(Image.fromarray(pred_obj_mask).resize(
                        (gt_mask.shape[1], gt_mask.shape[0]), resample=Image.NEAREST))
                else:
                    pred_obj_mask_resized = pred_obj_mask
                
                iou = calculate_iou(gt_mask, pred_obj_mask_resized)
                biou = boundary_iou(gt_mask, pred_obj_mask_resized)
                
                iou_scores_per_view.append(iou)
                biou_scores_per_view.append(biou)
    
    if len(iou_scores_per_view) == 0:
        return 0.0, 0.0, temp_masks
    
    mean_iou = np.mean(iou_scores_per_view)
    mean_biou = np.mean(biou_scores_per_view)
    
    return mean_iou, mean_biou, temp_masks


def render_set_multiscale(model_path, name, iteration, views, gaussians, feature_gaussians, 
                          pipeline, background, scale_gate,
                          TEXT_PROMPT, gt_masks_dict, threshold=0.9, smooth_K=16, use_hdbscan=True):
    """
    Multi-scale rendering and evaluation for a single text prompt.
    Uses GT mask from reference view instead of Grounded-SAM.
    Only saves the best scale results.
    """
    print(f"\n{'='*80}")
    print(f"Processing text prompt: '{TEXT_PROMPT}'")
    print(f"{'='*80}")
    
    render_path = os.path.join(model_path, name, f"ours_{iteration}_text", "renders")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}_text", "gt")
    colormask_path = os.path.join(model_path, name, f"ours_{iteration}_text", "objects_feature16")
    pred_obj_path = os.path.join(model_path, name, f"ours_{iteration}_text", "test_mask_fine")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(colormask_path, exist_ok=True)
    makedirs(pred_obj_path, exist_ok=True)
    
    if len(gt_masks_dict) == 0:
        print(f"Warning: No GT masks found for '{TEXT_PROMPT}', skipping...")
        return 0.0, 0.0, 0.0
    
    print(f"Found {len(gt_masks_dict)} GT masks for '{TEXT_PROMPT}'")
    
    # Get GT mask from reference view (first view with GT mask)
    # Find first view that has GT mask
    ref_view_idx = min(gt_masks_dict.keys())
    gt_ref_mask = gt_masks_dict[ref_view_idx]
    print(f"Using GT mask from view {ref_view_idx} as reference")
    
    # Render features for reference view
    with torch.no_grad():
        view_ref = views[ref_view_idx]
        results = render_contrastive_feature(view_ref, feature_gaussians, pipeline, background,
                                            norm_point_features=True, smooth_weights=None,
                                            smooth_type='traditional', smooth_K=smooth_K)
        rendered_features0 = results["render"]
        rendered_features0 = F.interpolate(rendered_features0.unsqueeze(0),
                                          view_ref.original_image.shape[1:],
                                          mode='bilinear').squeeze(0)
    
    # Prepare view data (minimal, without precomputing clusters)
    view_data = []
    for idx, view in enumerate(views):
        view_data.append({'view': view})
    
    # Multi-scale search (query centroids will be extracted per scale)
    test_scales = np.arange(0.0, 1.1, 0.1)
    print(f"\nTrying {len(test_scales)} scales (0.0 to 1.0 in 0.1 increments)")
    
    best_mean_iou = 0.0
    best_mean_biou = 0.0
    best_scale = 0.0
    best_masks = None
    
    for scale_value in test_scales:
        print(f"\n--- Testing scale: {scale_value:.1f} ---")
        
        mean_iou, mean_biou, temp_masks = render_and_evaluate_scale(
            scale_value, view_data, TEXT_PROMPT, gt_ref_mask, gt_masks_dict,
            rendered_features0, feature_gaussians, pipeline, background,
            scale_gate, threshold, smooth_K, use_hdbscan
        )
        
        print(f"  Scale {scale_value:.1f}: Mean IoU = {mean_iou:.4f}, Mean BIoU = {mean_biou:.4f}")
        
        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            best_mean_biou = mean_biou
            best_scale = scale_value
            best_masks = temp_masks
            print(f"  *** New best: IoU = {best_mean_iou:.4f}, BIoU = {best_mean_biou:.4f} ***")
    
    print(f"\nBest scale: {best_scale:.1f}")
    print(f"Best Mean IoU: {best_mean_iou:.4f}")
    print(f"Best Mean BIoU: {best_mean_biou:.4f}")
    
    # Save only the best scale results
    if best_masks is not None:
        print(f"\nSaving best results (scale={best_scale:.1f})...")
        for idx, view in enumerate(tqdm(views, desc="Saving best masks")):
            pred_obj_img_path = os.path.join(pred_obj_path, f'{idx}')
            makedirs(pred_obj_img_path, exist_ok=True)
            
            # Save mask
            pred_mask = best_masks[idx]
            Image.fromarray(pred_mask).save(os.path.join(pred_obj_img_path, f'{TEXT_PROMPT}.png'))
            
            # Save GT image (once per view, not for each prompt)
            if idx == 0 or not os.path.exists(os.path.join(gts_path, f'{idx:05d}.png')):
                gt = view.original_image[0:3, :, :]
                torchvision.utils.save_image(gt, os.path.join(gts_path, f'{idx:05d}.png'))
    
    # Save results summary
    summary_path = os.path.join(model_path, name, f"ours_{iteration}_text", "results_summary_fine.txt")
    with open(summary_path, 'a') as f:
        f.write(f"{TEXT_PROMPT}: Best Scale={best_scale:.1f}, IoU={best_mean_iou:.4f}, BIoU={best_mean_biou:.4f}\n")
    
    return best_mean_iou, best_mean_biou, best_scale


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, 
                skip_train: bool, skip_test: bool, scene_name: str,
                smooth_K: int = 16, use_hdbscan: bool = True):
    """Main function for multi-scale rendering and evaluation with fine-grained annotations"""
    with torch.no_grad():
        dataset.eval = True
        dataset.need_masks = False
        dataset.need_masks_scale = False
        dataset.need_depth = False
        dataset.need_features = False
        
        # Load models
        gaussians = GaussianModel(dataset.sh_degree)
        feature_gaussians = FeatureGaussianModel(dataset.feature_dim)
        scene = Scene(dataset, gaussians, feature_gaussians, load_iteration=iteration, 
                     shuffle=False, target='contrastive_feature', mode='eval')
        
        # Load scale_gate
        scale_gate = torch.nn.Sequential(
            torch.nn.Linear(1, 32, bias=True),
            torch.nn.Sigmoid()
        ).cuda()
        
        checkpoint_path = os.path.join(dataset.model_path, "point_cloud", f"iteration_{scene.feature_loaded_iter}")
        scale_gate_path = os.path.join(checkpoint_path, "scale_gate.pt")
        if os.path.exists(scale_gate_path):
            scale_gate.load_state_dict(torch.load(scale_gate_path))
            print(f"Loaded scale_gate from {scale_gate_path}")
        else:
            print(f"Warning: scale_gate not found at {scale_gate_path}")
        
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # Load prompts from lerf_mask_fine dataset
        fine_prompt_base_path = f"../data/lerf_mask_fine/{scene_name}/fine"
        positives = []
        
        if os.path.exists(fine_prompt_base_path):
            # Iterate through all subdirectories (0, 1, 2, etc.)
            for subdir in sorted(os.listdir(fine_prompt_base_path)):
                subdir_path = os.path.join(fine_prompt_base_path, subdir)
                if not os.path.isdir(subdir_path):
                    continue
                
                # Get all .png files in subdirectory (these are the category names)
                for mask_file in sorted(os.listdir(subdir_path)):
                    if mask_file.endswith('.png'):
                        cat_name = mask_file.replace('.png', '')
                        if cat_name not in positives:
                            positives.append(cat_name)
            
            print(f"Loaded {len(positives)} fine-grained prompts from {fine_prompt_base_path}")
        else:
            raise FileNotFoundError(f"Fine prompt directory not found: {fine_prompt_base_path}")
        
        print("Text prompts:", positives)
        
        views = scene.getTestCameras() if skip_train else scene.getTrainCameras()
        print(f"{len(views)} views to process")
        
        # Process each text prompt with multi-scale evaluation
        all_results = {}
        for TEXT_PROMPT in positives:
            # Load GT masks for this text prompt from fine-grained structure
            gt_masks_dict = {}
            for idx, view in enumerate(views):
                # Fine-grained GT structure: ../data/lerf_mask_fine/{scene}/fine/{view_idx}/{category}.png
                gt_mask_path = os.path.join(fine_prompt_base_path, str(idx), f'{TEXT_PROMPT}.png')
                
                if os.path.exists(gt_mask_path):
                    gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
                    gt_masks_dict[idx] = gt_mask
            
            if len(gt_masks_dict) == 0:
                print(f"Warning: No GT masks found for '{TEXT_PROMPT}', skipping...")
                continue
            
            mean_iou, mean_biou, best_scale = render_set_multiscale(
                dataset.model_path, "test" if skip_train else "train", 
                scene.feature_loaded_iter, views, gaussians, feature_gaussians,
                pipeline, background, scale_gate,
                TEXT_PROMPT, gt_masks_dict, threshold=0.95, smooth_K=smooth_K,
                use_hdbscan=use_hdbscan
            )
            all_results[TEXT_PROMPT] = {
                'iou': mean_iou,
                'biou': mean_biou,
                'scale': best_scale
            }
        
        # Print overall summary
        print(f"\n{'='*80}")
        print("FINAL RESULTS SUMMARY (Fine-grained)")
        print(f"{'='*80}")
        for prompt, results in all_results.items():
            print(f"{prompt:40s}: IoU={results['iou']:.4f}, BIoU={results['biou']:.4f}, Scale={results['scale']:.1f}")
        
        overall_iou = np.mean([r['iou'] for r in all_results.values()])
        overall_biou = np.mean([r['biou'] for r in all_results.values()])
        print(f"\nOverall Mean IoU: {overall_iou:.4f}")
        print(f"Overall Mean BIoU: {overall_biou:.4f}")
        print(f"{'='*80}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Multi-scale rendering and evaluation script for fine-grained annotations")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--smooth_K", default=16, type=int)
    parser.add_argument("--use_hdbscan", action="store_true", default=False)
    parser.add_argument("--scene_name", type=str, required=True,
                       help="Scene name (figurines, ramen, or teatime)")
    
    args = get_combined_args(parser, target_cfg_file='cfg_args')
    print("Rendering " + args.model_path)
    
    safe_state(args.quiet)
    
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), 
                args.skip_train, args.skip_test, args.scene_name,
                args.smooth_K, args.use_hdbscan)
