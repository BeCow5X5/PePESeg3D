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
import sys
import math
import uuid
import random

import numpy as np
import torch
from random import randint
from tqdm import tqdm

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_refinement import parallel_gaussian_splitting
from gaussian_renderer import render, render_with_cream
from lpipsPyTorch import lpips
from scene import Scene, GaussianModel
from scene.cameras import CamerasWrapper
from utils.depth_utils import compute_scale_and_shift
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

def set_seed(seed):
    """
    Set random seeds for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, decompose_from_arg=None, depth_from_arg=None):
    first_iter = 0

    dataset.need_masks = True
    dataset.need_masks_scale = False
    dataset.need_depth = True

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    decompose_from = decompose_from_arg if decompose_from_arg is not None else 5000
    decompose_until = None

    train_cameras = scene.getTrainCameras()
    viewpoint_stack = train_cameras.copy()
    p3d_training_cameras = CamerasWrapper(viewpoint_stack)
    fov_cameras, c2ws = p3d_training_cameras.p3d_cameras
    cam_index_stack = list(range(len(train_cameras)))
    
    if len(viewpoint_stack) > 100:
        decompose_until = decompose_from + 200
    else:
        decompose_until = decompose_from + 2*len(viewpoint_stack)
    
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    depth_from = depth_from_arg if depth_from_arg is not None else 1000
    depth_until = 30000
    original_gaussian_count = None

    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not cam_index_stack:
            cam_index_stack = list(range(len(train_cameras)))

        cam_idx = cam_index_stack.pop(randint(0, len(cam_index_stack) - 1))
        viewpoint_cam = train_cameras[cam_idx]
        fov_camera = fov_cameras[cam_idx]
        
        with torch.no_grad():
            dense_map = viewpoint_cam.original_masks.cuda().float()
            viewpoint_cam.feature_height, viewpoint_cam.feature_width = viewpoint_cam.image_height, viewpoint_cam.image_width
    
        if (iteration - 1) == debug_from:
            pipe.debug = True

        if iteration >= depth_from and iteration <= depth_until:
            render_pkg = render_with_cream(viewpoint_cam, gaussians, pipe, bg_color=background)
            image, viewspace_point_tensor, visibility_filter, radii, ddepth = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
            )
            
            # Depth weight is annealed with a reverse sigmoid: strong early (fix geometry)
            w_start = 0.05
            w_end   = 0.01
            steepness = 15.0
            center    = 0.5
            progress = (iteration - depth_from) / (depth_until - depth_from)
            sigmoid_factor = 1.0 / (1.0 + math.exp(steepness * (progress - center)))
            current_global_weight = w_end + (w_start - w_end) * sigmoid_factor
            current_local_weight = current_global_weight * 0.25

            rel_depth = viewpoint_cam.depth.to(ddepth.device)   # [1, H, W]
            valid_mask = torch.ones_like(rel_depth, dtype=torch.bool, device=ddepth.device)

            flat_rel = rel_depth[valid_mask].view(-1)
            q_low, q_high = 0.01, 0.99
            low_val  = torch.quantile(flat_rel, q_low)
            high_val = torch.quantile(flat_rel, q_high)
            inlier_mask = (rel_depth >= low_val) & (rel_depth <= high_val)
            global_mask = valid_mask & inlier_mask
            if global_mask.sum() < 10:
                global_mask = valid_mask

            scale, shift = compute_scale_and_shift(ddepth, rel_depth, global_mask)
            scale = torch.abs(scale)
            aligned_depth = scale.view(-1, 1, 1) * ddepth + shift.view(-1, 1, 1)
            diff_global = (aligned_depth - rel_depth)[global_mask]
            depth_global_loss = current_global_weight * diff_global.abs().mean()

            unique_mask_ids = torch.unique(dense_map)
            unique_mask_ids = unique_mask_ids[unique_mask_ids > 0]

            depth_local_loss = torch.tensor(0.0, device=ddepth.device)

            if unique_mask_ids.numel() > 0:
                sampled_mask_id = unique_mask_ids[
                    torch.randint(0, len(unique_mask_ids), (1,)).item()
                ]
                mask_pixels = (dense_map == sampled_mask_id).nonzero(as_tuple=False)
                num_mask_pixels = mask_pixels.shape[0]
                patch_size = int(np.sqrt(num_mask_pixels))                
                center_idx = torch.randint(0, mask_pixels.shape[0], (1,)).item()
                center_y, center_x = mask_pixels[center_idx]

                H, W = ddepth.shape[1], ddepth.shape[2]
                half_patch = patch_size

                y_start = max(0, center_y - half_patch)
                y_end   = min(H, center_y + half_patch + 1)
                x_start = max(0, center_x - half_patch)
                x_end   = min(W, center_x + half_patch + 1)

                rendered_patch = ddepth[:, y_start:y_end, x_start:x_end]
                target_patch   = rel_depth[:, y_start:y_end, x_start:x_end]
                rendered_patch_flat = rendered_patch.reshape(1, -1)
                target_patch_flat   = target_patch.reshape(1, -1)
                flat_target_patch = target_patch_flat.view(-1)

                if flat_target_patch.numel() > 0:
                    plow  = torch.quantile(flat_target_patch, q_low)
                    phigh = torch.quantile(flat_target_patch, q_high)
                    patch_inlier_mask = (target_patch_flat >= plow) & (target_patch_flat <= phigh)
                    if patch_inlier_mask.sum() < 5:
                        patch_inlier_mask = torch.ones_like(target_patch_flat, dtype=torch.bool, device=ddepth.device)
                else:
                    patch_inlier_mask = torch.ones_like(target_patch_flat, dtype=torch.bool, device=ddepth.device)

                patch_mask = patch_inlier_mask
                patch_scale, patch_shift = compute_scale_and_shift(
                    rendered_patch_flat.unsqueeze(0),
                    target_patch_flat.unsqueeze(0),
                    patch_mask.unsqueeze(0),
                )
                patch_scale = torch.abs(patch_scale)
                aligned_patch = patch_scale * rendered_patch_flat + patch_shift

                diff_local = (aligned_patch - target_patch_flat)[patch_inlier_mask]
                depth_local_loss = current_local_weight * diff_local.abs().mean()

            depth_loss = depth_global_loss + depth_local_loss

        else:
            render_pkg = render_with_cream(viewpoint_cam, gaussians, pipe, bg_color=background)
            image, viewspace_point_tensor, visibility_filter, radii, ddepth = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
            )
            depth_loss = torch.tensor(0.0, device="cuda")        
        
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss += depth_loss
        
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            if torch.isnan(loss):
                print(f"[Error] NaN detected in loss at iteration {iteration}")
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(iteration, l1_loss, testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            ## Gaussian Decomposition (Parallel with Splitting + Surface-Aware Sampling) ##
            with torch.no_grad():
                # Avoid decomposition only on densification iterations
                if (iteration < decompose_until and 
                    iteration > decompose_from and
                    iteration % opt.densification_interval != 0):
                    
                    # Record original gaussian count at first decomposition
                    if original_gaussian_count is None:
                        original_gaussian_count = gaussians.get_xyz.shape[0]
                        print(f"[ITER {iteration}] Recording original gaussian count: {original_gaussian_count}")
                    
                    # Create filter to exclude newly added gaussians
                    current_count = gaussians.get_xyz.shape[0]
                    original_gaussians_filter = torch.zeros(current_count, dtype=torch.bool, device="cuda")
                    original_gaussians_filter[:original_gaussian_count] = True  # Only original gaussians
                    
                    single_dense_map = dense_map

                    # Resize single_dense_map to match image size if needed
                    mask_h, mask_w = single_dense_map.shape[0], single_dense_map.shape[1]
                    img_h, img_w = viewpoint_cam.image_height, viewpoint_cam.image_width
                    if mask_h != img_h or mask_w != img_w:
                        print(f"[ITER {iteration}] Resizing single_dense_map from ({mask_h}, {mask_w}) to ({img_h}, {img_w})")
                        single_dense_map = torch.nn.functional.interpolate(
                            single_dense_map.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
                            size=(img_h, img_w),
                            mode='nearest'
                        ).squeeze(0).squeeze(0)  # (H, W)
                    
                    # Get unique mask IDs from single dense map
                    unique_mask_ids = torch.unique(single_dense_map)
                    unique_mask_ids = unique_mask_ids[unique_mask_ids > 0]  # Remove background (0)
                    
                    if len(unique_mask_ids) > 0:
                        print(f"[ITER {iteration}] Parallel Decomposition: Processing {len(unique_mask_ids)} mask ids simultaneously")
                        
                        gcenters_in_cam_coord = fov_camera.get_world_to_view_transform().transform_points(gaussians.get_xyz)
                        gcenters_z_cam = gcenters_in_cam_coord[..., 2] + 0.
                        gcenters_z_depth = p3d_training_cameras.get_points_depth_in_depth_map(fov_camera, ddepth, gcenters_in_cam_coord, cam_idx)
                        surface_dist = torch.abs(gcenters_z_cam - gcenters_z_depth)
                        
                        total_valid = int(visibility_filter.sum())
                        num_to_keep = max(1, int(total_valid * 0.1)) if total_valid > 0 else 0
                        surface_aware_filter = torch.zeros_like(visibility_filter, dtype=torch.bool)

                        if total_valid > 0 and num_to_keep > 0:
                            valid_distances = surface_dist[visibility_filter]
                            valid_indices = torch.where(visibility_filter)[0]

                            if valid_distances.numel() <= num_to_keep:
                                surface_aware_filter = visibility_filter.clone()
                            else:
                                _, topk_indices = torch.topk(valid_distances, num_to_keep, largest=False)
                                selected_indices = valid_indices[topk_indices]
                                surface_aware_filter[selected_indices] = True

                        combined_filter = surface_aware_filter & original_gaussians_filter
                        original_count = gaussians.get_xyz.shape[0]
                        gaussians = parallel_gaussian_splitting(
                            gaussians, 
                            viewpoint_cam, 
                            single_dense_map,
                            split_threshold=0.7,
                            surface_filter=combined_filter
                        )
                        new_count = gaussians.get_xyz.shape[0]
                        added_gaussians = new_count - original_count
                        
                        print(f"[ITER {iteration}] Decomposition completed: {original_count} → {new_count} gaussians (+{added_gaussians})")

                    else:
                        print(f"[ITER {iteration}] Decomposition: No valid masks found in single dense map")


            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def training_report(iteration, l1_loss, testing_iterations, scene: Scene, renderFunc, renderArgs):
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))

        print("[ITER {}] Gaussians: {}".format(iteration, scene.gaussians.get_xyz.shape[0]))
        torch.cuda.empty_cache()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if hasattr(args, 'output_folder') and args.output_folder:
            args.model_path = args.output_folder
        else:
            if os.getenv('OAR_JOB_ID'):
                unique_str=os.getenv('OAR_JOB_ID')
            else:
                unique_str = str(uuid.uuid4())
            args.model_path = os.path.join("./output/", unique_str[0:10])
        
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=np.random.randint(10000, 20000))
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 15_000, 22_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--output_folder", type=str, default=None, help="Custom output folder path")
    parser.add_argument("--seed", default=0, type=int,
                        help="Random seed. Default 0 matches safe_state(), i.e. the published runs.")
    parser.add_argument('--decompose_from', type=int, default=None, help='Iteration to start decomposition (overrides default)')
    parser.add_argument('--depth_from', type=int, default=None, help='Iteration to start depth losses (overrides default)')
    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)


    # Initialize system state (RNG)
    safe_state(args.quiet)
    set_seed(args.seed)
    print("Random seed: {}".format(args.seed))

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    prepare_output_and_logger(args)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.decompose_from, args.depth_from)

    # All done
    print("\nTraining complete.")
