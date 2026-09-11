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
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, fetchPly
from scene.gaussian_model import GaussianModel
from scene.gaussian_model_ff import FeatureGaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel
    feature_gaussians : FeatureGaussianModel

    # target: feature, seg, scene
    def __init__(self, args : ModelParams, gaussians : GaussianModel=None, feature_gaussians: FeatureGaussianModel=None, load_iteration=None, feature_load_iteration=None, shuffle=True, resolution_scales=[1.0], init_from_3dgs_pcd=False, target='scene', mode='train', sample_rate = 1.0):
        
        self.model_path = args.model_path
        self.loaded_iter = None # for pretrained gaussians ckpt
        self.feature_loaded_iter = None # for feature gaussians ckpt
        self.gaussians = gaussians
        self.feature_gaussians = feature_gaussians

        if load_iteration:
            if load_iteration == -1: # default, load lastest pretrained gaussians
                if mode == 'train':
                    if target == 'contrastive_feature':
                        mask_dir = "sam_masks"
                        self.feature_loaded_iter = None
                        self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="scene")
                    elif target == 'scene':
                        mask_dir = "single_dense_maps"
                        self.feature_loaded_iter = None
                        self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="scene")
                    else:
                        assert False and "Unknown target!"
                elif mode == 'eval':
                    if target == 'contrastive_feature':
                        mask_dir = "sam_masks"
                        self.feature_loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target=target) if (feature_load_iteration is None or feature_load_iteration == -1) else feature_load_iteration

                        self.loaded_iter = -1 if gaussians is None else searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target='scene')
                    elif target == 'scene':
                        mask_dir = "single_dense_maps"
                        self.feature_gaussians = None
                        self.feature_loaded_iter = None
                        self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="scene")
                    else:
                        assert False and "Unknown target!"
            else: # specific iteration
                self.loaded_iter = load_iteration
                if mode == 'train':
                    if target == 'contrastive_feature':
                        mask_dir = "sam_masks"
                        self.feature_loaded_iter = None
                    elif target == 'scene':
                        mask_dir = "single_dense_maps"
                        self.feature_loaded_iter = None
                    else:
                        assert False and "Unknown target!"
                elif mode == 'eval':
                    if target == 'contrastive_feature':
                        mask_dir = "sam_masks"
                        self.feature_loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target=target) if (feature_load_iteration is None or feature_load_iteration == -1) else feature_load_iteration
                        self.loaded_iter = -1
                    elif target == 'scene':
                        mask_dir = "single_dense_maps"
                        self.feature_gaussians = None
                        self.feature_loaded_iter = None
                    else:
                        assert False and "Unknown target!"

            print("Loading trained model at iteration {}, {}".format(self.loaded_iter, self.feature_loaded_iter))
        else: # default for train_scene_gd
            mask_dir = "single_dense_maps"
            print("load_iteration is None, training from scratch!")
            
        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            print(f"Allow Camera Principle Point Shift: {args.allow_principle_point_shift}")
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path,
                args.images,
                args.eval,
                need_masks=args.need_masks,
                mask_dir=mask_dir,
                need_masks_scale=args.need_masks_scale,
                need_depth=args.need_depth,
                sample_rate=sample_rate,
                allow_principle_point_shift=args.allow_principle_point_shift,
                replica='replica' in args.model_path,
                train_split=args.train_split
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path,
                args.white_background,
                args.eval
            )
        else:
            assert False, "Could not recognize scene type!"
        
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        # Load scene gaussians
        if self.loaded_iter and self.gaussians is not None:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                    "scene_point_cloud.ply"))
        # Initialize scene gaussians from Colmap point cloud
        elif self.gaussians is not None:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        # Load feature gaussians
        if self.feature_loaded_iter and self.feature_gaussians is not None:
            if target == 'contrastive_feature':
                if mode == 'train':
                    self.feature_gaussians.load_ply_from_3dgs(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "scene_point_cloud.ply"))
                elif mode == 'eval':
                    print("HELLLLOO")
                    self.feature_gaussians.load_ply(os.path.join(self.model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.feature_loaded_iter),
                                                        "contrastive_feature_point_cloud.ply"))

        # Initialize feature gaussians from scene gaussians
        elif self.feature_gaussians is not None:
            if target == 'contrastive_feature':
                if mode == 'train':
                    self.feature_gaussians.load_ply_from_3dgs(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "scene_point_cloud.ply"))
                elif mode == 'eval':
                    self.feature_gaussians.load_ply(os.path.join(self.model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.feature_loaded_iter),
                                                        "contrastive_feature_point_cloud.ply"))
            else:
                print("Initialize feature gaussians from Colmap point cloud")
                self.feature_gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)


    def save(self, iteration, target='scene'):
        assert target != 'feature' and "Please use save_feature() to save feature gaussians!"
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, target+"_point_cloud.ply"))

    def save_mask(self, iteration, id = 0):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_mask(os.path.join(point_cloud_path, f"seg_point_cloud_{id}.npy"))

    def save_feature(self, iteration, target = 'contrastive_feature', smooth_weights = None, smooth_type = None, smooth_K = 16):
        assert self.feature_gaussians is not None and target == 'contrastive_feature'
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.feature_gaussians.save_ply(os.path.join(point_cloud_path, f"{target}_point_cloud.ply"), smooth_weights, smooth_type, smooth_K)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]