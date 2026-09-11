import time
import os

import torch
import pytorch3d.ops
from plyfile import PlyData, PlyElement
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from argparse import ArgumentParser, Namespace
import cv2

from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel, FeatureGaussianModel
from gaussian_renderer import render, render_contrastive_feature
from copy import deepcopy
from utils.sh_utils import SH2RGB
import hdbscan

def get_combined_args(parser : ArgumentParser, model_path, target_cfg_file = None):
    cmdlne_string = ['--model_path', model_path]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)
    
    if target_cfg_file is None:
        if args_cmdline.target == 'seg':
            target_cfg_file = "seg_cfg_args"
        elif args_cmdline.target == 'scene' or args_cmdline.target == 'xyz':
            target_cfg_file = "cfg_args"
        elif args_cmdline.target == 'feature' or args_cmdline.target == 'coarse_seg_everything' or args_cmdline.target == 'contrastive_feature' :
            target_cfg_file = "feature_cfg_args"

    try:
        cfgfilepath = os.path.join(model_path, target_cfg_file)
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file found: {}".format(cfgfilepath))
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v

    return Namespace(**merged_dict)

from sklearn.preprocessing import QuantileTransformer

# Borrowed from GARField, but modified
def get_quantile_func(scales: torch.Tensor, distribution="normal"):
    """
    Use 3D scale statistics to normalize scales -- use quantile transformer.
    """
    scales = scales.flatten()

    scales = scales.detach().cpu().numpy()
    print(scales.max(), '?')

    # Calculate quantile transformer
    quantile_transformer = QuantileTransformer(output_distribution=distribution)
    quantile_transformer = quantile_transformer.fit(scales.reshape(-1, 1))

    
    def quantile_transformer_func(scales):
        scales_shape = scales.shape

        scales = scales.reshape(-1,1)
        
        return torch.Tensor(
            quantile_transformer.transform(scales.detach().cpu().numpy())
        ).to(scales.device).reshape(scales_shape)

    return quantile_transformer_func, quantile_transformer

import os

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Process paths and iteration count.")
    parser.add_argument('--source_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--iteration', type=int, default=10000)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    print(f"Source Path: {args.source_path}")
    print(f"Output Path: {args.output_path}")
    print(f"FeatureGaussian Iteration: {args.iteration}")

    FEATURE_DIM = 32
    MODEL_PATH = args.source_path # 30000
    OUTPUT_PATH = args.output_path
    FEATURE_GAUSSIAN_ITERATION = args.iteration
    SCALE_GATE_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/scale_gate.pt')
    FEATURE_PCD_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/contrastive_feature_point_cloud.ply')
    SCENE_PCD_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/scene_point_cloud.ply')

    # Load multiple scale gates (0.0 to 1.0)
    # Simple scale gate with improved initialization for diversity
    scale_gate = torch.nn.Sequential(
        torch.nn.Linear(1, 32, bias=True),
        torch.nn.Sigmoid()  # Keep Sigmoid for bounded output [0, 1]
    )
    scale_gate = scale_gate.cuda()
    scale_gate.load_state_dict(torch.load(SCALE_GATE_PATH))
    scale_gate.eval()

    scale_values = [0.0, 0.5, 1.0]
    print(f"Evaluating with {len(scale_values)} different scales: {scale_values}")
    
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument('--target', default='scene', type=str)

    args = get_combined_args(parser, MODEL_PATH)

    dataset = model.extract(args)

    dataset.need_depth = False
    dataset.need_features = False

    # To obtain mask scales
    dataset.need_masks = False
    dataset.need_masks_scale = False

    scene_gaussians = GaussianModel(dataset.sh_degree)

    feature_gaussians = FeatureGaussianModel(FEATURE_DIM)
    scene = Scene(dataset, scene_gaussians, feature_gaussians, load_iteration=-1, feature_load_iteration=FEATURE_GAUSSIAN_ITERATION, shuffle=False, mode='eval', target='contrastive_feature')

    cameras = scene.getTrainCameras()
    print("There are",len(cameras),"views in the dataset.")
    
    # Process all cameras
    for camera_id in range(len(cameras)):
        print(f"\nProcessing camera {camera_id}...")
        
        view = deepcopy(cameras[camera_id])

        view.feature_height, view.feature_width = view.image_height, view.image_width
        img = view.original_image * 255
        print(f"Camera {camera_id} image shape:", img.shape)
        img = img.permute([1,2,0]).detach().cpu().numpy().astype(np.uint8)

        bg_color = [0 for i in range(FEATURE_DIM)]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        rendered_feature = render_contrastive_feature(view, feature_gaussians, pipeline.extract(args), background, norm_point_features=True, smooth_type = None)['render']
        feature_h, feature_w = rendered_feature.shape[-2:]

        # Create camera-specific folder
        camera_output_path = f'./output_images/{OUTPUT_PATH}/camera_{camera_id}'
        if not os.path.exists(camera_output_path):
            os.makedirs(camera_output_path)

        img_pil = Image.fromarray(img)
        img_pil.save(f'{camera_output_path}/original_image.png')

        with torch.no_grad():
            # Evaluate with multiple scale values
            for idx, scale_val in enumerate(scale_values):
                scale = torch.tensor([scale_val]).cuda()
                
                # Use corresponding gate for this scale
                gates = scale_gate(scale).squeeze(0)  # [32]
                
                print(f"Scale {scale_val:.1f} - gates shape:", gates.shape)
                print(f"rendered_feature shape:", rendered_feature.shape)
                
                feature_with_scale = rendered_feature * gates.unsqueeze(-1).unsqueeze(-1)  # [32, H, W]
                scale_conditioned_feature = feature_with_scale.permute([1,2,0])  # [H, W, 32]
                print(f"Camera {camera_id} scale {scale_val:.1f} conditioned feature shape:", scale_conditioned_feature.shape)

                # Save scale conditioned feature
                scale_output_path = f'{camera_output_path}/scale_{scale_val:.1f}'
                os.makedirs(scale_output_path, exist_ok=True)
                img_pil = Image.fromarray((scale_conditioned_feature[:,:,:3].detach().cpu().numpy()*255).astype(np.uint8))
                img_pil.save(f'{scale_output_path}/scale_conditioned_feature.png')

                # Query-based similarity for this scale
                query_index_orig = (400, 600)
                query_index = (
                    int(query_index_orig[0] / view.image_height * view.feature_height),
                    int(query_index_orig[1] / view.image_width * view.feature_width),
                )

                normed_features = torch.nn.functional.normalize(scale_conditioned_feature, dim = -1, p = 2)
                query_feature = normed_features[query_index[0], query_index[1]]

                similarity = torch.einsum('C,HWC->HW', query_feature, normed_features)

                img_pil = Image.fromarray(similarity.detach().cpu().numpy()*255)
                img_pil = img_pil.convert("RGB")
                img_pil.save(f'{scale_output_path}/similarity.png')

                img_pil = Image.fromarray(similarity.detach().cpu().numpy() > 0.75)
                img_pil.save(f'{scale_output_path}/similarity_threshold.png')

                # HDBSCAN clustering with two methods
                H, W = scale_conditioned_feature.shape[0], scale_conditioned_feature.shape[1]
                use_random_sampling = False  # Set to True for random sampling, False for downsampling
                
                if use_random_sampling:
                    # Method 1: Random sampling (better feature preservation)
                    max_samples = 16384
                    print(f"[HDBSCAN Scale {scale_val:.1f}] Using random sampling with max_samples={max_samples}")
                    
                    # Reshape: [H, W, C] -> [H*W, C]
                    F = scale_conditioned_feature.reshape(-1, FEATURE_DIM)
                    N = F.shape[0]
                    
                    # Random sampling for speed
                    if N > max_samples:
                        indices = torch.randperm(N, device=F.device)[:max_samples]
                        F_sampled = F[indices]
                        sampled_positions = indices  # Store positions for label assignment
                    else:
                        F_sampled = F
                        sampled_positions = torch.arange(N, device=F.device)
                    
                    # Normalize features
                    F_sampled_norm = torch.nn.functional.normalize(F_sampled, dim=-1, p=2)
                    
                    # HDBSCAN clustering
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=10, 
                        min_samples=5,
                        cluster_selection_epsilon=0.01,
                        metric='euclidean'
                    )
                    cluster_labels_sampled = clusterer.fit_predict(F_sampled_norm.detach().cpu().numpy())
                    
                    # Assign labels to all pixels based on nearest sampled pixel
                    labels_upsampled = np.full((H * W,), -1, dtype=np.int32)
                    labels_upsampled[sampled_positions.cpu().numpy()] = cluster_labels_sampled
                    
                    # For unsampled pixels, find nearest labeled pixel (batch processing to avoid OOM)
                    unlabeled_mask = labels_upsampled == -1
                    if unlabeled_mask.any():
                        F_norm_all = torch.nn.functional.normalize(F, dim=-1, p=2)
                        unlabeled_indices = torch.where(torch.from_numpy(unlabeled_mask).to(F.device))[0]
                        
                        # Compute similarity to sampled features in batches
                        F_labeled = F_sampled_norm  # [N_sampled, C]
                        batch_size = 4096  # Process 4096 unlabeled pixels at a time
                        nearest_labels = []
                        
                        for i in range(0, len(unlabeled_indices), batch_size):
                            batch_indices = unlabeled_indices[i:i+batch_size]
                            F_unlabeled_batch = F_norm_all[batch_indices]  # [batch_size, C]
                            
                            # Find nearest labeled feature for this batch
                            sim = torch.mm(F_unlabeled_batch, F_labeled.T)  # [batch_size, N_sampled]
                            nearest_labeled_idx = sim.argmax(dim=1)  # [batch_size]
                            nearest_labels.append(cluster_labels_sampled[nearest_labeled_idx.cpu().numpy()])
                        
                        # Assign labels
                        nearest_labels = np.concatenate(nearest_labels)
                        labels_upsampled[unlabeled_indices.cpu().numpy()] = nearest_labels
                    
                    labels_upsampled = labels_upsampled.reshape(H, W)
                    print(f"Camera {camera_id} Scale {scale_val:.1f} (random sampling) unique labels:", np.unique(labels_upsampled))
                    
                else:
                    # Method 2: Fixed downsampling (faster but may lose details)
                    H_down = 128
                    W_down = 128
                    
                    print(f"[HDBSCAN Scale {scale_val:.1f}] Downsampling from {H}x{W} to {H_down}x{W_down} for clustering")
                    
                    downsampled_features = torch.nn.functional.interpolate(
                        scale_conditioned_feature.permute([2,0,1]).unsqueeze(0), 
                        (H_down, W_down), 
                        mode='bilinear',  align_corners=False
                    ).squeeze()
                    cluster_normed_features = torch.nn.functional.normalize(downsampled_features, dim = 0, p = 2).permute([1,2,0])

                    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, cluster_selection_epsilon=0.01)
                    cluster_labels = clusterer.fit_predict(cluster_normed_features.reshape([-1, cluster_normed_features.shape[-1]]).detach().cpu().numpy())
                    labels = cluster_labels.reshape([cluster_normed_features.shape[0], cluster_normed_features.shape[1]])
                    print(f"Camera {camera_id} Scale {scale_val:.1f} (downsampling) unique labels:", np.unique(labels))

                    # Upsample labels back to original size
                    labels_upsampled = cv2.resize(labels.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
                
                # Cluster centers based on HDBSCAN labels
                cluster_centers = []
                for label in np.unique(labels_upsampled):
                    if label == -1:  # skip noise
                        continue
                    mask = (labels_upsampled == label)
                    cluster_features = normed_features[mask]
                    cluster_center = cluster_features.mean(dim=0)
                    cluster_centers.append(cluster_center)
                
                if len(cluster_centers) == 0:
                    print(f"[Warning Scale {scale_val:.1f}] No valid clusters found, skipping segmentation")
                    continue
                    
                cluster_centers = torch.stack(cluster_centers)

                label_to_color = np.random.rand(200, 3)
                segmentation_res = torch.einsum('nc,hwc->hwn', cluster_centers.cuda(), normed_features)

                segmentation_res_idx = segmentation_res.argmax(dim = -1)
                colored_labels = (label_to_color[segmentation_res_idx.cpu().numpy().astype(np.int8)] * 255).astype(np.uint8)

                img_pil = Image.fromarray(colored_labels)
                img_pil.save(f'{scale_output_path}/colored_labels.png')

                # Scale conditioned point features for this scale
                point_features = feature_gaussians.get_point_features
                scale_conditioned_point_features = point_features * gates.unsqueeze(0)
                normed_scale_conditioned_point_features = torch.nn.functional.normalize(scale_conditioned_point_features, dim = -1, p = 2)

                similarities = torch.einsum('C,NC->N', query_feature.cuda(), normed_scale_conditioned_point_features)
                similarities[similarities < 0.7] = 0

                bg_color = [0 for i in range(FEATURE_DIM)]
                background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                rendered_similarities = render(view, scene_gaussians, pipeline.extract(args), background, override_color=similarities.unsqueeze(-1).repeat([1,3]))['render']

                img_pil = Image.fromarray(rendered_similarities.permute([1,2,0])[:,:,0].detach().cpu().numpy() > 0.75)
                img_pil.save(f'{scale_output_path}/rendered_similarities.png')

        try:
            scene_gaussians.roll_back()
        except:
            pass
        scene_gaussians.segment(similarities > 0.95)

        # save the segmentation
        name = f'precomputed_mask_camera_{camera_id}'
        os.makedirs('./segmentation_res', exist_ok=True)
        torch.save(similarities > 0.75, f'./segmentation_res/{name}.pt')

        bg_color = [1 for i in range(FEATURE_DIM)]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        rendered_segmented_image = render(view, scene_gaussians, pipeline.extract(args), background)['render']

        img_pil = Image.fromarray((rendered_segmented_image.permute([1,2,0]).detach().cpu().numpy()* 255).astype(np.uint8))
        img_pil.save(f'{camera_output_path}/rendered_segmented_image.png')

        scene_gaussians.roll_back()
        
        print(f"Camera {camera_id} processing completed. Results saved to {camera_output_path}")
    
    print(f"\nAll cameras processed successfully! Results saved in ./output_images/{OUTPUT_PATH}/camera_{{id}} folders")
