import math
import time

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as func

# ============================================================================
# CORE PROJECTION AND ASSIGNMENT FUNCTIONS
# ============================================================================

def project_to_2d(viewpoint_camera, points3D):
    """
    Project 3D points to 2D image plane
    
    Args:
        viewpoint_camera: Camera object with projection matrices
        points3D: [N, 3] tensor of 3D points
        
    Returns:
        point_image: [N, 2] tensor of 2D image coordinates
    """
    full_matrix = viewpoint_camera.full_proj_transform  # w2c @ K 
    # project to image plane
    points3D = F.pad(input=points3D, pad=(0, 1), mode='constant', value=1)
    p_hom = (points3D @ full_matrix).transpose(0, 1)  # N, 4 -> 4, N   -1 ~ 1
    p_w = 1.0 / (p_hom[-1, :] + 0.0000001)
    p_proj = p_hom[:3, :] * p_w

    h = viewpoint_camera.image_height
    w = viewpoint_camera.image_width

    point_image = 0.5 * ((p_proj[:2] + 1) * torch.tensor([w, h]).unsqueeze(-1).to(p_proj.device) - 1) # image plane
    point_image = point_image.detach().clone()
    point_image = torch.round(point_image.transpose(0, 1))

    return point_image

def multi_mask_assignment(xyz, viewpoint_camera, dense_mask):
    """
    Multi-mask parallel assignment for gaussians
    
    Args:
        xyz: [N, 3] tensor of gaussian centers
        viewpoint_camera: Camera object
        dense_mask: [H, W] dense mask tensor with different mask IDs
        
    Returns:
        point_masks: [N] tensor of mask ID assignments for each gaussian
        mask_gaussian_indices: dict {mask_id: tensor of gaussian indices}
    """
    w2c_matrix = viewpoint_camera.world_view_transform
    xyz_homo = F.pad(input=xyz, pad=(0, 1), mode='constant', value=1)
    p_view = (xyz_homo @ w2c_matrix[:, :3]).transpose(0, 1)
    depth = p_view[-1, :].detach().clone()
    valid_depth = depth >= 0

    h, w = viewpoint_camera.image_height, viewpoint_camera.image_width
    
    if dense_mask.shape[0] != h or dense_mask.shape[1] != w:
        dense_mask = func.resize(dense_mask.unsqueeze(0), (h, w), antialias=True).squeeze(0).long()
    else:
        dense_mask = dense_mask.long()
    
    point_image = project_to_2d(viewpoint_camera, xyz).long()
    
    valid_x = (point_image[:, 0] >= 0) & (point_image[:, 0] < w)
    valid_y = (point_image[:, 1] >= 0) & (point_image[:, 1] < h)
    valid_mask = valid_x & valid_y & valid_depth
    
    point_masks = torch.full((point_image.shape[0],), -1, device=xyz.device)
    point_masks[valid_mask] = dense_mask[point_image[valid_mask, 1], point_image[valid_mask, 0]]
    
    unique_mask_ids = torch.unique(point_masks)
    unique_mask_ids = unique_mask_ids[unique_mask_ids > 0]  # Remove background and invalid
    
    mask_gaussian_indices = {}
    for mask_id in unique_mask_ids:
        mask_gaussian_indices[mask_id.item()] = torch.where(point_masks == mask_id)[0]
    
    return point_masks, mask_gaussian_indices


# ============================================================================
# COVARIANCE MATRIX UTILITIES
# ============================================================================

def compute_conv3d(conv3d):
    """
    Convert 6D covariance representation to 3x3 matrix
    
    Args:
        conv3d: [N, 6] tensor of covariance parameters
        
    Returns:
        complete_conv3d: [N, 3, 3] tensor of covariance matrices
    """
    complete_conv3d = torch.zeros((conv3d.shape[0], 3, 3), device=conv3d.device)
    complete_conv3d[:, 0, 0] = conv3d[:, 0]
    complete_conv3d[:, 1, 0] = conv3d[:, 1]
    complete_conv3d[:, 0, 1] = conv3d[:, 1]
    complete_conv3d[:, 2, 0] = conv3d[:, 2]
    complete_conv3d[:, 0, 2] = conv3d[:, 2]
    complete_conv3d[:, 1, 1] = conv3d[:, 3]
    complete_conv3d[:, 2, 1] = conv3d[:, 4]
    complete_conv3d[:, 1, 2] = conv3d[:, 4]
    complete_conv3d[:, 2, 2] = conv3d[:, 5]

    return complete_conv3d

def conv2d_matrix(gaussians, viewpoint_camera, indices_mask, device):
    """
    Compute 2D covariance matrices for specified gaussians
    
    Args:
        gaussians: GaussianModel object
        viewpoint_camera: Camera object
        indices_mask: tensor of gaussian indices to process
        device: torch device
        
    Returns:
        conv2d_matrix: [N, 2, 2] tensor of 2D covariance matrices
    """
    # 3d convariance matrix
    conv3d = gaussians.get_covariance(scaling_modifier=1)[indices_mask]
    conv3d_matrix = compute_conv3d(conv3d).to(device)

    w2c = viewpoint_camera.world_view_transform
    mask_xyz = gaussians.get_xyz[indices_mask]
    pad_mask_xyz = F.pad(input=mask_xyz, pad=(0, 1), mode='constant', value=1)
    t = pad_mask_xyz @ w2c[:, :3]   # N, 3
    height = viewpoint_camera.image_height
    width = viewpoint_camera.image_width
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_x = width / (2.0 * tanfovx)
    focal_y = height / (2.0 * tanfovy)
    lim_xy = torch.tensor([1.3 * tanfovx, 1.3 * tanfovy]).to(device)
    t[:, :2] = torch.clip(t[:, :2] / t[:, 2, None], -1. * lim_xy, lim_xy) * t[:, 2, None]
    J_matrix = torch.zeros((mask_xyz.shape[0], 3, 3)).to(device)
    J_matrix[:, 0, 0] = focal_x / t[:, 2]
    J_matrix[:, 0, 2] = -1 * (focal_x * t[:, 0]) / (t[:, 2] * t[:, 2])
    J_matrix[:, 1, 1] = focal_y / t[:, 2]
    J_matrix[:, 1, 2] = -1 * (focal_y * t[:, 1]) / (t[:, 2] * t[:, 2])
    W_matrix = w2c[:3, :3]  # 3,3
    T_matrix = (W_matrix @ J_matrix.permute(1, 2, 0)).permute(2, 0, 1) # N,3,3

    conv2d_matrix = torch.bmm(T_matrix.permute(0, 2, 1), torch.bmm(conv3d_matrix, T_matrix))[:, :2, :2]

    return conv2d_matrix


def conv2d_matrix_batch(gaussians, viewpoint_camera, indices_list, device):
    """
    Batch computation of 2D covariance matrices for multiple mask groups
    
    Args:
        gaussians: GaussianModel object
        viewpoint_camera: Camera object
        indices_list: list of tensor indices for each mask group
        device: torch device
        
    Returns:
        conv2d_matrix: [total_N, 2, 2] tensor of 2D covariance matrices
    """
    if len(indices_list) == 0:
        return torch.empty(0, 2, 2, device=device)
    all_indices = torch.cat(indices_list)
    result = conv2d_matrix(gaussians, viewpoint_camera, all_indices, device)
    
    return result

## Gaussian Decomposition
# ============================================================================
# PARALLEL DECOMPOSITION AND SPLITTING FUNCTIONS
# ============================================================================

def compute_split_ratios_batch_vectorized(conv2d, points_xy, binary_mask, h, w, split_threshold, points_xyz=None, viewpoint_camera=None, gaussians=None, indices=None, max_samples=None):
    """
    Vectorized batch computation based on compute_ratios and update functions logic
    
    Args:
        conv2d: [N, 2, 2] tensor of 2D covariance matrices
        points_xy: [N, 2] tensor of 2D point coordinates
        binary_mask: [H, W] binary mask tensor
        h, w: image dimensions
        split_threshold: threshold for determining splits (not used in original logic)
        points_xyz: [N, 3] tensor of 3D point coordinates (optional, for 3D split positions)
        viewpoint_camera: Camera object (optional, for depth-aware splitting)
        gaussians: GaussianModel object (optional, for 3D covariance and scaling)
        indices: [N] tensor of gaussian indices (optional, for accessing 3D properties)
        max_samples: maximum number of samples per line for efficiency
        
    Returns:
        ratios: [N] tensor of coverage ratios (only for gaussians that cross mask boundary)
        needs_split: [N] boolean tensor indicating which gaussians need splitting
        split_directions: [N, 2] tensor of 2D split directions
        split_positions: [N, 3] tensor of 3D split positions
    """
    device = conv2d.device
    n_gaussians = conv2d.shape[0]
    
    eigvals, eigvecs = torch.linalg.eigh(conv2d)
    max_eigval, max_idx = torch.max(eigvals, dim=1)
    max_eigvec = torch.gather(eigvecs, dim=1, 
                        index=max_idx.unsqueeze(1).unsqueeze(2).repeat(1,1,2))
    
    # 3 sigma: calculate the coordinates of two vertices - same as compute_ratios
    long_axis = torch.sqrt(max_eigval) * 3
    max_eigvec = max_eigvec.squeeze(1)
    max_eigvec = max_eigvec / torch.norm(max_eigvec, dim=1).unsqueeze(-1)
    vertex1 = points_xy + 0.5 * long_axis.unsqueeze(1) * max_eigvec
    vertex2 = points_xy - 0.5 * long_axis.unsqueeze(1) * max_eigvec
    
    vertex1 = torch.clip(vertex1, torch.tensor([0, 0], device=device), torch.tensor([w-1, h-1], device=device))
    vertex2 = torch.clip(vertex2, torch.tensor([0, 0], device=device), torch.tensor([w-1, h-1], device=device))    
    vertex1_xy = torch.round(vertex1).long()
    vertex2_xy = torch.round(vertex2).long()
    vertex1_label = binary_mask[vertex1_xy[:, 1], vertex1_xy[:, 0]]
    vertex2_label = binary_mask[vertex2_xy[:, 1], vertex2_xy[:, 0]]
    
    # Find Gaussians that cross the mask boundary (need adjustment) - same as compute_ratios
    boundary_mask = (vertex1_label.long() ^ vertex2_label.long()).bool()
    ratios = torch.zeros(n_gaussians, device=device)
    needs_split = torch.zeros(n_gaussians, dtype=torch.bool, device=device)
    split_directions = torch.zeros(n_gaussians, 2, device=device)
    split_positions = torch.zeros(n_gaussians, 3, device=device)    
    boundary_indices = torch.where(boundary_mask)[0]
    
    if len(boundary_indices) == 0:
        return ratios, needs_split, split_directions, split_positions
    
    boundary_vertex1 = vertex1[boundary_indices]
    boundary_vertex2 = vertex2[boundary_indices]
    boundary_vertex1_label = vertex1_label[boundary_indices]
    boundary_vertex2_label = vertex2_label[boundary_indices]
    boundary_max_eigvec = max_eigvec[boundary_indices]
    boundary_long_axis = long_axis[boundary_indices]
    
    sign_direction = boundary_vertex1_label - boundary_vertex2_label
    direction_vector = boundary_max_eigvec * sign_direction.unsqueeze(-1)
    n_boundary = len(boundary_indices)
    
    line_lengths = torch.norm(boundary_vertex2 - boundary_vertex1, dim=1)
    if max_samples is None:
        # Use adaptive sampling like original method
        sample_counts = torch.clamp(line_lengths.int(), min=5, max=200)
        max_samples_needed = sample_counts.max().item()
    else:
        # Use fixed sampling
        sample_counts = torch.full((n_boundary,), max_samples, device=device, dtype=torch.int)
        max_samples_needed = max_samples
    
    t_vals = torch.linspace(0, 1, max_samples_needed, device=device)
    t_vals = t_vals.unsqueeze(0).expand(n_boundary, -1)
    
    # Vectorized line interpolation: [n_boundary, max_samples_needed, 2]
    sample_points = boundary_vertex1.unsqueeze(1) + t_vals.unsqueeze(-1) * (boundary_vertex2 - boundary_vertex1).unsqueeze(1)
    sample_points = torch.round(sample_points).long()
    sample_points[:, :, 0] = torch.clamp(sample_points[:, :, 0], 0, w-1)
    sample_points[:, :, 1] = torch.clamp(sample_points[:, :, 1], 0, h-1)
    flat_indices = sample_points[:, :, 1] * w + sample_points[:, :, 0]
    flat_mask = binary_mask.flatten()
    in_mask = flat_mask[flat_indices]
    
    # Compute ratios using only the required number of samples for each boundary gaussian
    boundary_ratios = torch.zeros(n_boundary, device=device)
    for i in range(n_boundary):
        actual_samples = sample_counts[i].item()
        boundary_ratios[i] = in_mask[i, :actual_samples].float().mean()
    
    ratios[boundary_indices] = boundary_ratios
    needs_split[boundary_indices] = True
    split_directions[boundary_indices] = direction_vector
    
    if points_xyz is not None and gaussians is not None and indices is not None:
        boundary_actual_indices = indices[boundary_indices] if indices is not None else boundary_indices
        boundary_3d_cov = gaussians.get_covariance(scaling_modifier=1)[boundary_actual_indices]
        boundary_3d_cov_matrices = compute_conv3d(boundary_3d_cov)
        
        eigvals_3d, eigvecs_3d = torch.linalg.eigh(boundary_3d_cov_matrices)
        max_eigval_3d, max_idx_3d = torch.max(eigvals_3d, dim=1)
        max_eigvec_3d = torch.gather(eigvecs_3d, dim=2, 
                            index=max_idx_3d.unsqueeze(1).unsqueeze(2).repeat(1,3,1)).squeeze(2)  # [n_boundary, 3]
        
        max_eigvec_3d = max_eigvec_3d / (torch.norm(max_eigvec_3d, dim=1).unsqueeze(-1) + 1e-8)        
        long_axis_3d = torch.sqrt(max_eigval_3d) * 3
        split_distances = 0.5 * (1 - boundary_ratios) * long_axis_3d
        max_eigvec_2d = project_to_2d(viewpoint_camera, max_eigvec_3d)
        
        sign_directions_3d = torch.sum(max_eigvec_2d * direction_vector, dim=1)
        sign_directions_3d = torch.where(sign_directions_3d > 0, 1, -1)
        
        boundary_split_positions = max_eigvec_3d * split_distances.unsqueeze(-1) * sign_directions_3d.unsqueeze(-1)
        split_positions[boundary_indices] = boundary_split_positions
        
    else:
        # Fallback: use 2D-based calculation for boundary gaussians
        split_distances = (1.0 - boundary_ratios) * boundary_long_axis * 0.5
        split_positions[boundary_indices, :2] = direction_vector * split_distances.unsqueeze(-1)
        split_positions[boundary_indices, 2] = 0.0  # No z-component for 2D fallback

    return ratios, needs_split, split_directions, split_positions


def parallel_gaussian_splitting(gaussians, viewpoint_camera, dense_mask, split_threshold=0.7, surface_filter=None):
    """
    Parallel gaussian decomposition with splitting capability
    
    Args:
        gaussians: GaussianModel object
        viewpoint_camera: Camera object
        dense_mask: [H, W] dense mask tensor with different mask IDs
        split_threshold: threshold for splitting gaussians (default: 0.7)
        surface_filter: Optional [N] boolean tensor to filter gaussians close to surface
        
    Returns:
        gaussians: Updated GaussianModel with split gaussians added
    """
    start_total = time.time()
    
    # Step 1: Initial setup and mask assignment
    start_step = time.time()
    xyz = gaussians.get_xyz
    point_masks, mask_gaussian_indices = multi_mask_assignment(xyz, viewpoint_camera, dense_mask)
    
    if len(mask_gaussian_indices) == 0:
        return gaussians
    
    # Step 2: Collect all indices for batch processing
    start_step = time.time()
    all_indices = []
    mask_to_batch_idx = {}
    start_idx = 0

    for mask_id, indices in mask_gaussian_indices.items():
        mask_to_batch_idx[mask_id] = (start_idx, start_idx + len(indices))
        all_indices.append(indices)
        start_idx += len(indices)
    
    if len(all_indices) == 0:
        return gaussians
    
    # Step 3: Batch compute 2D covariance matrices
    start_step = time.time()
    conv2d_batch = conv2d_matrix_batch(gaussians, viewpoint_camera, all_indices, device=xyz.device)
    
    # Step 4: Project to 2D
    start_step = time.time()
    point_image = project_to_2d(viewpoint_camera, xyz)
    
    new_xyz_list = []
    new_features_list = []
    new_scaling_list = []
    new_rotation_list = []
    new_opacity_list = []
    updated_indices = []
    updated_scaling = []
    updated_xyz = []  # For position updates of existing gaussians
    
    height, width = viewpoint_camera.image_height, viewpoint_camera.image_width
    
    # Step 5: Process each mask
    start_step = time.time()
    batch_offset = 0
    mask_processing_times = []
    mask_preprocessing_times = []
    
    processed_gaussians_tensor = torch.zeros(len(xyz), dtype=torch.bool, device=xyz.device)
    
    for mask_id, indices in mask_gaussian_indices.items():
        mask_start = time.time()
        if len(indices) == 0:
            continue
        
        original_indices = indices
        unprocessed_mask = ~processed_gaussians_tensor[indices]
        
        if not unprocessed_mask.any():
            continue  # All gaussians already processed
        
        indices = indices[unprocessed_mask]
        
        if surface_filter is not None:
            surface_mask = surface_filter[indices]
            indices = indices[surface_mask]
            
            if len(indices) == 0:
                continue  # No surface gaussians for this mask
        
        binary_mask = (dense_mask == mask_id).float()
        start_batch, end_batch = mask_to_batch_idx[mask_id]
        
        original_batch_size = end_batch - start_batch
        if len(indices) < original_batch_size:
            batch_indices_range = torch.arange(start_batch, end_batch, device=indices.device)
            original_indices_batch = torch.cat(all_indices)[batch_indices_range]            
            batch_mask = torch.isin(original_indices_batch, indices)
            conv2d = conv2d_batch[start_batch:end_batch][batch_mask]
        else:
            conv2d = conv2d_batch[start_batch:end_batch]
        
        mask_point_image = point_image[indices]
        mask_point_xyz = xyz[indices]
        
        mask_preprocessing_times.append(time.time() - mask_start)

        ratios, needs_split, split_directions, split_positions = compute_split_ratios_batch_vectorized(
            conv2d, mask_point_image, binary_mask, height, width, split_threshold, 
            mask_point_xyz, viewpoint_camera, gaussians, indices
        )
        
        mask_processing_times.append(time.time() - mask_start)        
        split_mask = needs_split
        
        if split_mask.sum() > 0:
            split_indices = indices[split_mask]
            split_ratios = ratios[split_mask]
            split_dirs = split_directions[split_mask]
            split_pos = split_positions[split_mask]
            
            orig_scaling = gaussians.get_scaling[split_indices]
            orig_xyz = gaussians.get_xyz[split_indices]
            orig_features = gaussians.get_features[split_indices]
            orig_rotation = gaussians.get_rotation[split_indices] 
            orig_opacity = gaussians.get_opacity[split_indices]
            
            # Update existing gaussians to represent the INSIDE part (following update function logic)
            # Move towards mask interior and scale according to ratio
            inside_xyz = orig_xyz + split_pos  # Move towards mask interior
            inside_scaling = orig_scaling * split_ratios.unsqueeze(-1) * 0.8
            
            updated_indices.extend(split_indices.cpu().tolist())
            updated_scaling.append(inside_scaling)
            updated_xyz.append(inside_xyz)
            
            outside_ratio = (1.0 - split_ratios).unsqueeze(-1)
            outside_scaling = orig_scaling * outside_ratio * 0.8
            outside_xyz = orig_xyz - split_pos  # Move away from mask interior
            
            new_xyz_list.append(outside_xyz)
            new_features_list.append(orig_features)
            new_scaling_list.append(outside_scaling)
            new_rotation_list.append(orig_rotation)
            new_opacity_list.append(orig_opacity)
        
        processed_gaussians_tensor[indices] = True
    
    
    # Step 7: Apply updates to existing gaussians
    start_step = time.time()
    if updated_indices:
        gaussians._xyz = gaussians._xyz.detach().clone()
        gaussians._scaling = gaussians._scaling.detach().clone()
        
        updated_indices = torch.tensor(updated_indices, device=xyz.device)
        updated_scaling_cat = torch.cat(updated_scaling, dim=0)
        updated_xyz_cat = torch.cat(updated_xyz, dim=0)
        
        gaussians._scaling[updated_indices] = gaussians.scaling_inverse_activation(updated_scaling_cat)
        gaussians._xyz[updated_indices] = updated_xyz_cat  # Update positions too
    
    # Step 8: Add new gaussians using densification_postfix for proper optimizer handling
    start_step = time.time()
    if new_xyz_list:
        new_xyz = torch.cat(new_xyz_list, dim=0)
        new_features = torch.cat(new_features_list, dim=0) 
        new_scaling = torch.cat(new_scaling_list, dim=0)
        new_rotation = torch.cat(new_rotation_list, dim=0)
        new_opacity = torch.cat(new_opacity_list, dim=0)
        
        # Use densification_postfix to properly handle optimizer state extension
        gaussians.densification_postfix(
            new_xyz,
            new_features[:, :1, :],  # features_dc
            new_features[:, 1:, :],  # features_rest
            gaussians.inverse_opacity_activation(new_opacity),  # Convert back to raw opacity
            gaussians.scaling_inverse_activation(new_scaling),  # Convert back to raw scaling
            new_rotation
        )
        
    
    
    return gaussians