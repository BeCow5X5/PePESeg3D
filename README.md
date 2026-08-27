# PePESeg3D

Official implementation of

**PePESeg3D: Perception Prior Enhances Multi-Scale Segmentation for 3D Gaussian Splatting**
BMVC 2026

---

PePESeg3D injects 2D perception priors — SAM masks and monocular depth — into *both* stages of a
multi-scale 3DGS segmentation pipeline:

- **PePE Reconstruction** refines Gaussian primitives onto segmentation boundaries and regularises
  geometry with a monocular-depth prior, producing a semantically aligned backbone.
- **PePE Contrastive Learning** learns the scale-aware feature field on that frozen geometry using
  scale-aware mask supervision, dense depth–colour perception cues, and view-consistent centroids.

The method reaches state-of-the-art multi-scale segmentation and scene reconstruction on the
SPIn-NeRF, LERF-Mask and NVOS benchmarks.

## Status

**Code and pretrained checkpoints will be released here.** The release will include the two-stage
training pipeline, the preprocessing scripts for the perception priors, and the evaluation code for
all reported benchmarks.

## Acknowledgements

Built on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting),
[SAGA](https://github.com/Jumpat/SegAnyGAussians),
[Segment Anything](https://github.com/facebookresearch/segment-anything),
[Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) and
[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO).
