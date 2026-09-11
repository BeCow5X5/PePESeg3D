import os
import cv2
import sys
import torch
import random
import importlib
import numpy as np
import gaussian_renderer

from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from matplotlib import pyplot as plt
from segment_anything import (SamAutomaticMaskGenerator,
                              sam_model_registry)
from sklearn.preprocessing import QuantileTransformer
from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel, FeatureGaussianModel

importlib.reload(gaussian_renderer)

ALLOW_PRINCIPLE_POINT_SHIFT = False


def prepare_output_and_logger(args):    
    if not hasattr(args, 'model_path') or not args.model_path:
        if hasattr(args, 'output_folder') and args.output_folder:
            args.model_path = args.output_folder
        else:
            # For extract_masks_and_scales, we don't need a model_path
            # Just skip creating cfg_args if no model_path is provided
            print("No model_path provided, skipping config file creation")
            return
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def generate_grid_index(depth):
    """Generate grid indices for depth map"""
    h, w = depth.shape
    grid = torch.meshgrid([torch.arange(h), torch.arange(w)])
    grid = torch.stack(grid, dim=-1)
    return grid


def get_quantile_transformer(scales, distribution="uniform"):
    """Create quantile transformer to normalize scales to 0.0-1.0"""
    scales_flat = scales.flatten().detach().cpu().numpy()
    
    # Remove invalid scales (0.0, nan, inf)
    valid_mask = np.isfinite(scales_flat) & (scales_flat > 0)
    scales_valid = scales_flat[valid_mask]
    
    if len(scales_valid) == 0:
        print("Warning: No valid scales found")
        return None
        
    quantile_transformer = QuantileTransformer(output_distribution=distribution)
    quantile_transformer.fit(scales_valid.reshape(-1, 1))
    
    def transform_scales(input_scales):
        input_flat = input_scales.flatten().detach().cpu().numpy()
        valid_mask = np.isfinite(input_flat) & (input_flat > 0)
        
        result = np.zeros_like(input_flat)
        if np.any(valid_mask):
            result[valid_mask] = quantile_transformer.transform(input_flat[valid_mask].reshape(-1, 1)).flatten()
        
        return torch.from_numpy(result.reshape(input_scales.shape)).to(input_scales.device)
    
    return transform_scales


def create_single_dense_map(masks):
    """Flatten the SAM masks of one view into a single 2D ID map, fine masks on top."""
    if len(masks) == 0:
        return None
    
    H, W = masks.shape[-2:]
    
    dense_map = torch.zeros((H, W), dtype=torch.int16, device=masks.device)
    mask_counter = 1
    available_ids = []
    
    # Paint largest first so that finer masks are drawn last and stay on top.
    mask_areas = masks.sum(dim=(1, 2))
    sort_indices = torch.argsort(mask_areas, descending=True)  # largest first
    sorted_masks = masks[sort_indices]
    
    overlap_threshold = 0.7
    
    for mask in sorted_masks:
        mask_gpu = mask.bool()
        
        existing_ids = dense_map[mask_gpu]
        unique_ids = torch.unique(existing_ids[existing_ids > 0])

        for existing_id in unique_ids:
            id_pixels = (dense_map == existing_id)
            
            intersection = torch.logical_and(id_pixels, mask_gpu).sum().float()
            existing_id_area = id_pixels.sum().float()
            
            if existing_id_area > 0:
                overlap_ratio = intersection / existing_id_area
                if overlap_ratio >= overlap_threshold:
                    dense_map[id_pixels] = 0
                    available_ids.append(existing_id.item())
        
        if available_ids:
            current_id = available_ids.pop(0)
            dense_map[mask_gpu] = current_id
        else:
            dense_map[mask_gpu] = mask_counter
            mask_counter += 1

    return dense_map

def extract_sam_masks(args):
    """Extract SAM masks from images"""
    print("=== STEP 1: Extracting SAM segment everything masks ===")
    
    # Initialize SAM
    print("Initializing SAM...")
    model_type = args.sam_arch
    sam = sam_model_registry[model_type](checkpoint=args.sam_checkpoint_path).to('cuda')
    
    # Configure mask generator
    # SAGA Version
    # mask_generator = SamAutomaticMaskGenerator(
    # )

    # OmniSeg3DGS Version
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        points_per_batch=64,      # 256
        pred_iou_thresh=.88,
        stability_score_thresh=.95,  # default: 0.95, LLFF: 0.9
        stability_score_offset=1,
        box_nms_thresh=.7,
        crop_n_layers=0,  # default: 0, LLFF: 1
        crop_nms_thresh=.7,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=128
    )

    # Determine image directory with downsample support
    downsample_manually = False
    if args.downsample == 1 or args.downsample_type == 'mask':
        IMAGE_DIR = os.path.join(args.source_path, 'images')
    else:
        IMAGE_DIR = os.path.join(args.source_path, f'images_{args.downsample}')
        if not os.path.exists(IMAGE_DIR):
            IMAGE_DIR = os.path.join(args.source_path, 'images')
            downsample_manually = True
            print(f"No downsampled images found, will downsample manually by factor {args.downsample}")
    
    assert os.path.exists(IMAGE_DIR), "Please specify a valid source path"
    
    # Create output directory
    OUTPUT_DIR = os.path.join(args.source_path, 'sam_masks')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Extracting SAM segment everything masks (downsample={args.downsample}, type={args.downsample_type})...")
    
    # Process each image
    for path in tqdm(sorted(os.listdir(IMAGE_DIR))):
        name = path.split('.')[0]
        img = cv2.imread(os.path.join(IMAGE_DIR, path))
        
        # Downsample image manually if needed
        if downsample_manually:
            img = cv2.resize(img, 
                           dsize=(img.shape[1] // args.downsample, img.shape[0] // args.downsample),
                           fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
        
        masks = mask_generator.generate(img)
        mask_list = []
        
        for m in masks:
            m_score = torch.from_numpy(m['segmentation']).float().to('cuda')

            # Downsample mask if needed
            if args.downsample_type == 'mask' and args.downsample > 1:
                original_h, original_w = img.shape[0], img.shape[1]
                target_h = original_h // args.downsample
                target_w = original_w // args.downsample
                m_score = torch.nn.functional.interpolate(
                    m_score.unsqueeze(0).unsqueeze(0), 
                    size=(target_h, target_w),
                    mode='bilinear', 
                    align_corners=False
                ).squeeze()
                m_score = (m_score >= 0.5).bool()

            if len(m_score.unique()) < 2:
                continue
            else:
                mask_list.append(m_score.bool())
    
        if mask_list:
            masks = torch.stack(mask_list, dim=0)
            torch.save(masks, os.path.join(OUTPUT_DIR, name+'.pt'))
    
    print(f"SAM masks saved to {OUTPUT_DIR}")


def generate_single_dense_maps(args, dataset):
    """Generate single dense map from masks without scale information"""
    print("=== STEP 2 : Generating single dense maps without scale information ===")
    
    scene_gaussians = GaussianModel(dataset.sh_degree)

    if not hasattr(dataset, 'source_path') or not dataset.source_path:
        dataset.source_path = args.source_path
        print(f"Using args.source_path as dataset.source_path: {dataset.source_path}")

    scene = Scene(dataset, scene_gaussians)

    sam_masks_dir = os.path.join(dataset.source_path, 'sam_masks')
    if not os.path.exists(sam_masks_dir):
        sam_masks_dir = os.path.join(args.source_path, 'sam_masks')
        assert os.path.exists(sam_masks_dir), f"SAM masks not found in {dataset.source_path} or {args.source_path}. Please run SAM mask extraction first."
        print(f"Using sam_masks from: {sam_masks_dir}")
    
    OUTPUT_DIR = os.path.join(args.source_path, 'single_dense_maps')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    VIS_OUTPUT_DIR = os.path.join(args.source_path, 'single_dense_maps_vis')
    os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)
    
    cameras = scene.getTrainCameras()
    
    print("Generating single dense map for each camera view...")
    for view in tqdm(cameras):
        mask_file = os.path.join(dataset.source_path, 'sam_masks', view.image_name + '.pt')
        
        if not os.path.exists(mask_file):
            print(f"Warning: Missing mask file for {view.image_name}")
            continue
            
        masks = torch.load(mask_file)
        dense_map = create_single_dense_map(masks)
        
        if dense_map is not None:
            output_file = os.path.join(OUTPUT_DIR, f"{view.image_name}.pt")
            torch.save({'dense_map': dense_map}, output_file)
            
            # Save visualization
            vis_output_file = os.path.join(VIS_OUTPUT_DIR, f"{view.image_name}_single.png")
            visualize_dense_map(dense_map, vis_output_file, seed=42)
    
    print(f"Single dense maps saved to {OUTPUT_DIR}")
    print(f"Visualizations saved to {VIS_OUTPUT_DIR}")


def calculate_mask_scales(args, dataset, pipeline_params):
    """Calculate 3D scales for the extracted masks"""
    print("=== STEP 3: Calculating mask scales ===")
    
    feature_gaussians = None
    scene_gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, scene_gaussians, feature_gaussians, 
                 load_iteration=-1, feature_load_iteration=-1, 
                 shuffle=False, target='scene')

    assert os.path.exists(os.path.join(dataset.source_path, 'images')), "Please specify a valid image root."
    assert os.path.exists(os.path.join(dataset.source_path, 'sam_masks')), "Please run SAM mask extraction first."

    images_masks = {}
    for i, image_path in tqdm(enumerate(sorted(os.listdir(os.path.join(dataset.source_path, 'images'))))):
        mask_path = image_path.replace('jpg', 'pt').replace('JPG', 'pt').replace('png', 'pt')
        mask_file = os.path.join(dataset.source_path, 'sam_masks', mask_path)
        
        if os.path.exists(mask_file):
            masks = torch.load(mask_file) # N_mask, C
            images_masks[image_path.split('.')[0]] = masks.cpu().float()

    OUTPUT_DIR = os.path.join(args.source_path, 'mask_scales')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cameras = scene.getTrainCameras()
    background = torch.zeros(scene_gaussians.get_mask.shape[0], 3, device='cuda')

    print("Calculating scales for each camera view...")
    for it, view in tqdm(enumerate(cameras)):
        rendered_pkg = gaussian_renderer.render_with_cream(view, scene_gaussians, pipeline_params, background)
        depth = rendered_pkg['depth'].cpu().squeeze()

        if view.image_name not in images_masks:
            print(f"Warning: No masks found for {view.image_name}")
            continue
            
        corresponding_masks = images_masks[view.image_name]

        grid_index = generate_grid_index(depth)
        points_in_3D = torch.zeros(depth.shape[0], depth.shape[1], 3).cpu()
        points_in_3D[:, :, -1] = depth

        cx = depth.shape[1] / 2
        cy = depth.shape[0] / 2
        fx = cx / np.tan(view.FoVx / 2)
        fy = cy / np.tan(view.FoVy / 2)

        points_in_3D[:, :, 0] = (grid_index[:, :, 1] - cx) * depth / fx
        points_in_3D[:, :, 1] = (grid_index[:, :, 0] - cy) * depth / fy

        upsampled_mask = torch.nn.functional.interpolate(
            corresponding_masks.unsqueeze(1), mode='bilinear', 
            size=(depth.shape[0], depth.shape[1]), align_corners=False
        )

        eroded_masks = torch.conv2d(
            upsampled_mask.float(),
            torch.full((3, 3), 1.0).view(1, 1, 3, 3),
            padding=1,
        )
        eroded_masks = (eroded_masks >= 5).squeeze()

        scale = torch.zeros(len(corresponding_masks))
        for mask_id in range(len(corresponding_masks)):
            if mask_id < len(eroded_masks):
                point_in_3D_in_mask = points_in_3D[eroded_masks[mask_id] == 1]
                
                if len(point_in_3D_in_mask) > 0:
                    scale[mask_id] = (point_in_3D_in_mask.std(dim=0) * 2).norm()
                else:
                    scale[mask_id] = 0.0

        torch.save(scale, os.path.join(OUTPUT_DIR, view.image_name + '.pt'))

    print(f"Mask scales saved to {OUTPUT_DIR}")


def generate_random_colors(num_labels, seed=42):
    """Generate random colors for each label"""
    random.seed(seed)
    colors = []
    for i in range(num_labels):
        if i == 0:  # Background color (black)
            colors.append([0, 0, 0])
        else:
            colors.append([random.randint(0, 255) for _ in range(3)])
    return colors


def visualize_dense_map(dense_map, output_path, seed=42):
    """Visualize dense map with random colors and save as PNG"""
    # Get unique labels
    unique_labels = torch.unique(dense_map).cpu().numpy()
    print(unique_labels)
    num_labels = len(unique_labels)
    
    # Generate random colors
    colors = generate_random_colors(num_labels + 1, seed)  # +1 for safety
    
    H, W = dense_map.shape
    rgb_image = np.zeros((H, W, 3), dtype=np.uint8)
    
    for label in unique_labels:
        mask = (dense_map == label).cpu().numpy()
        label_idx = min(int(label), len(colors) - 1)
        rgb_image[mask] = colors[label_idx]
    
    Image.fromarray(rgb_image).save(output_path)


if __name__ == '__main__':
    parser = ArgumentParser(description="Extract SAM masks and calculate their 3D scales")
    
    # SAM-related arguments
    parser.add_argument("--sam_checkpoint_path", default='./third_party/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth', type=str)
    parser.add_argument("--sam_arch", default="vit_h", type=str)
    parser.add_argument("--downsample", default=1, type=int, help="Downsample factor for images")
    parser.add_argument("--downsample_type", default='image', type=str, choices=['image', 'mask'], 
                       help="Downsample then segment, or segment then downsample.")
    
    # Processing options
    parser.add_argument("--skip_mask_extraction", action="store_true", 
                       help="Skip SAM mask extraction if masks already exist")
    parser.add_argument("--skip_single_dense_map", action="store_true",
                       help="Skip single dense map generation")
    parser.add_argument("--skip_scale_calculation", action="store_true", 
                       help="Skip scale calculation")
    
    # Add model and pipeline params
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    
    # Additional arguments
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--segment", action="store_true")
    parser.add_argument('--idx', default=0, type=int)
    parser.add_argument('--precomputed_mask', default=None, type=str)
    parser.add_argument("-o", "--output_folder", type=str, default=None, help="Custom output folder path")
    
    args = parser.parse_args(sys.argv[1:])
    args.eval = False
    
    # Prepare output and logger (optional for this script)
    prepare_output_and_logger(args)
    
    print("=== Loaded Arguments ===", args)

    dataset = model.extract(args)
    dataset.need_masks = False
    dataset.need_depth = False
    print("Dataset loaded:", dataset)

    # Handle ALLOW_PRINCIPLE_POINT_SHIFT safely
    if hasattr(args, 'model_path') and args.model_path:
        dataset.allow_principle_point_shift = 'lerf' in args.model_path
    else:
        dataset.allow_principle_point_shift = ALLOW_PRINCIPLE_POINT_SHIFT

    # Create pipeline params once for main use
    pipeline_params = pipeline.extract(args)

    # Execute the pipeline
    try:
        if not args.skip_mask_extraction:
            extract_sam_masks(args)
        else:
            print("Skipping SAM mask extraction...")
        
        if not args.skip_single_dense_map:
            generate_single_dense_maps(args, dataset)
        else:
            print("Skipping single dense map generation...")

        if not args.skip_scale_calculation:
            calculate_mask_scales(args, dataset, pipeline_params)
        else:
            print("Skipping scale calculation...")
            
        print("=== Pipeline completed successfully! ===")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        raise