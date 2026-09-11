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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal, getProjectionMatrix
from pytorch3d.renderer import FoVPerspectiveCameras as P3DCameras
from pytorch3d.renderer.cameras import _get_sfm_calibration_matrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, cx = None, cy = None, masks = None, mask_scales = None, depth = None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.original_masks = masks
        self.mask_scales = mask_scales
        
        # Convert depth to CUDA tensor for optimization
        if depth is not None:
            # Convert numpy depth to preprocessed CUDA tensor
            depth_data = depth[:,:,0].astype(np.float32)
            depth_tensor = torch.from_numpy(depth_data).to(self.data_device, non_blocking=True)
            self.depth = (1.0 - depth_tensor / 255.0).unsqueeze(0)
        else:
            self.depth = None

        self.feature_width = self.image_width // 2
        self.feature_height = self.image_height // 2

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, cx=cx, cy=cy, w=self.image_width, h=self.image_height).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    @property
    def device(self):
        return self.world_view_transform.device
    
    def to(self, device):
        self.world_view_transform = self.world_view_transform.to(device)
        self.projection_matrix = self.projection_matrix.to(device)
        self.full_proj_transform = self.full_proj_transform.to(device)
        self.camera_center = self.camera_center.to(device)
        return self

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


# In SuGaR use this
class GSCamera(torch.nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                image_name, uid, semantic_feature,
                trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda", objects=None,
                image_height=None, image_width=None,
                ):
        """
        Args:
            colmap_id (int) : ID of the camera in the COLMAP reconstruction.
            R (np.array) : Rotation matrix.
            T (np.array) : Translation vector.
            FoVx (float) : Field of view in the x direction.
            FoVy (float) : Field of view in the y direction.
            image (np.array) : GT image.
            gt_alpha_mask (_type_): _description_
            image_name (_type_): _description_
            uid (_type_): _description_
            trans (_type_, optional): _description_. Defaults to np.array([0.0, 0.0, 0.0]).
            scale (float, optional): _description_. Defaults to 1.0.
            data_device (str, optional): _description_. Defaults to "cuda".
            image_height (_type_, optional): _description_. Defaults to None.
            image_width (_type_, optional): _description_. Defaults to None.
        """
        pass

def convert_camera_from_gs_to_pytorch3d(gs_cameras, device='cuda'):
    """
    From Gaussian Splatting camera parameters,
    computes R, T, K matrices and outputs pytorch3d-compatible camera object.

    Args:
        gs_cameras(List of GSCamera): List of Gaussian Splatting cameras. (result of cameraList_from_camInfos)
        device : Defaults to 'cuda'
    
    Returns:
        p3d_cameras: pytorch3d-compatible camera object.
    """
    N = len(gs_cameras)
    R = torch.Tensor(np.array([gs_camera.R for gs_camera in gs_cameras])).to(device)
    T = torch.Tensor(np.array([gs_camera.T for gs_camera in gs_cameras])).to(device)
    fx = torch.Tensor(np.array([fov2focal(gs_camera.FoVx, gs_camera.image_width) for gs_camera in gs_cameras])).to(device)
    fy = torch.Tensor(np.array([fov2focal(gs_camera.FoVy, gs_camera.image_height) for gs_camera in gs_cameras])).to(device)
    image_height = torch.tensor(np.array([gs_camera.image_height for gs_camera in gs_cameras]), dtype=torch.int).to(device)
    image_width = torch.tensor(np.array([gs_camera.image_width for gs_camera in gs_cameras]), dtype=torch.int).to(device)
    cx = image_width / 2
    cy = image_height / 2
    
    w2c = torch.zeros(N, 4, 4).to(device)
    w2c[:, :3, :3] = R.transpose(-1, -2)
    w2c[:, :3, 3] = T
    w2c[:, 3, 3] = 1

    c2w = w2c.inverse()
    c2w[:, :3, 1:3] *= -1 # flip y, z to align camera coordinate with world coordinate
    c2w = c2w[:, :3, :]

    image_size = torch.Tensor(
        [image_width[0], image_height[0]],
    )[
        None
    ].to(device)
    scale = image_size.min(dim=1, keepdim=True)[0] / 2.0
    c0 = image_size / 2.0
    p0_pytorch3d = ( # principal point of camera
        -(
            torch.Tensor((cx[0], cy[0]),)[None].to(device) - c0
        ) / scale
    )
    focal_pytorch3d = (
        torch.Tensor([fx[0], fy[0]])[None].to(device) / scale
    )
    K = _get_sfm_calibration_matrix(
        1, "cpu", focal_pytorch3d, p0_pytorch3d, orthographic=False
    )
    K = K.expand(N, -1, -1)

    # Convert the 3DGS extrinsics into the convention PyTorch3D expects.
    # PyTorch3D looks down +z, 3DGS looks down -z, hence the round trip through c2w.
    line = torch.Tensor([[0.0, 0.0, 0.0, 1.0]]).to(device).expand(N, -1, -1)
    cam2world = torch.cat([c2w, line], dim=1)
    world2cam = cam2world.inverse()
    R, T = world2cam.split([3, 1], dim=-1)
    R = R[:, :3].transpose(1, 2) * torch.Tensor([-1.0, 1.0, -1]).to(device)
    T = T.squeeze(2)[:, :3] * torch.Tensor([-1.0, 1.0, -1]).to(device)

    p3d_cameras = P3DCameras(device=device, R=R, T=T, K=K, znear=0.0001)
    
    return p3d_cameras, c2w
    

class CamerasWrapper:
    """
    Class to wrap Gaussian Splatting camera parameters
    and facilitates both usage and integration with PyTorch3D.
    """
    def __init__(
        self,
        gs_cameras,
        p3d_cameras=None,
        p3d_cameras_computed=False,
    ) -> None:    
        self.gs_cameras = gs_cameras

        self._p3d_cameras = p3d_cameras
        self._p3d_cameras_computed = p3d_cameras_computed

        device = gs_cameras[0].device
        N = len(gs_cameras)
        R = torch.Tensor(np.array([gs_camera.R for gs_camera in gs_cameras])).to(device)
        T = torch.Tensor(np.array([gs_camera.T for gs_camera in gs_cameras])).to(device)
        self.fx = torch.Tensor(np.array([fov2focal(gs_camera.FoVx, gs_camera.image_width) for gs_camera in gs_cameras])).to(device)
        self.fy = torch.Tensor(np.array([fov2focal(gs_camera.FoVy, gs_camera.image_height) for gs_camera in gs_cameras])).to(device)
        self.height = torch.tensor(np.array([gs_camera.image_height for gs_camera in gs_cameras]), dtype=torch.int).to(device)
        self.width = torch.tensor(np.array([gs_camera.image_width for gs_camera in gs_cameras]), dtype=torch.int).to(device)
        self.cx = self.width / 2
        self.cy = self.height / 2

        w2c = torch.zeros(N, 4, 4).to(device)  # already world-to-camera as stored on the 3DGS camera
        w2c[:, :3, :3] = R.transpose(-1, -2)
        w2c[:, :3, 3] = T
        w2c[:, 3, 3] = 1

        c2w = w2c.inverse()
        c2w[:, :3, 1:3] *= -1 # flip y, z to align camera coordinate with world coordinate
        c2w = c2w[:, :3, :]
        self.camera_to_worlds = c2w
        self.c2ws = None

    @property
    def device(self):
        return self.camera_to_worlds.device
    
    @property
    def p3d_cameras(self):
        if not self._p3d_cameras_computed:
            self._p3d_cameras, self.c2ws = convert_camera_from_gs_to_pytorch3d(
                self.gs_cameras,
            )
            self._p3d_cameras_computed = True
        return self._p3d_cameras, self.c2ws

    def to(self, device):
        self.camera_to_worlds = self.camera_to_worlds.to(device)
        self.fx = self.fx.to(device)
        self.fy = self.fy.to(device)
        self.cx = self.cx.to(device)
        self.cy = self.cy.to(device)
        self.width = self.width.to(device)
        self.height = self.height.to(device)
        
        for gs_camera in self.gs_cameras:
            gs_camera.to(device)

        if self._p3d_cameras_computed:
            self._p3d_cameras = self._p3d_cameras.to(device)

        return self

    def get_spatial_extent(self):
        """Returns the spatial extent of the cameras, computed as 
        the extent of the bounding box containing all camera centers.

        Returns:
            (float): Spatial extent of the cameras.
        """
        camera_centers = self.p3d_cameras.get_camera_center()
        avg_camera_center = camera_centers.mean(dim=0, keepdim=True)
        half_diagonal = torch.norm(camera_centers - avg_camera_center, dim=-1).max().item()

        radius = 1.1 * half_diagonal
        return radius
    
    # implement sugar.get_points_depth_in_depth_map without sugar class
    def get_points_depth_in_depth_map(self, fov_camera, depth, gaussian_centers_in_camera_space, idx):
        # Project the points with pytorch3d, then read the rendered depth at the pixel each falls on.
        depth_view = depth.unsqueeze(0)
        pts_projections = fov_camera.get_projection_transform().transform_points(gaussian_centers_in_camera_space)

        factor = -1 * min(self.height[idx], self.width[idx])
        pts_projections[..., 0] = factor / self.width[idx] * pts_projections[..., 0]
        pts_projections[..., 1] = factor / self.height[idx] * pts_projections[..., 1]
        pts_projections = pts_projections[..., :2].view(1, -1, 1, 2)

        map_z = torch.nn.functional.grid_sample(input=depth_view,
                                                grid=pts_projections,
                                                mode='bilinear',
                                                padding_mode='border'
                                                )[0, 0, :, 0]
        return map_z
