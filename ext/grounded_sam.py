import argparse
import os
import sys
import copy


import numpy as np
import torch
from PIL import Image

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt


from huggingface_hub import hf_hub_download

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model   

def show_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))





def grouned_sam_output(groundingdino_model, sam_predictor, TEXT_PROMPT, image, BOX_TRESHOLD = 0.3, TEXT_TRESHOLD = 0.45, device='cuda', filter_phrase=None, exclude_phrase=None):
    """
    Args:
        filter_phrase: Optional string to filter detected phrases. 
                      If provided, only boxes matching this phrase will be used.
                      Example: "noodles" will filter out "bowl" from "wavy noodles in bowl"
        exclude_phrase: Optional string to exclude from mask.
                       Regions detected with this phrase will be removed from final mask.
                       Example: "bowl" will remove bowl regions from the mask
    """
    image_source = image
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(Image.fromarray(image_source), None)

    boxes, logits, phrases = predict(
        model=groundingdino_model, 
        image=image, 
        caption=TEXT_PROMPT, 
        box_threshold=BOX_TRESHOLD, 
        text_threshold=TEXT_TRESHOLD
    )
    
    print(f"Original phrases detected: {phrases}")
    
    # Separate boxes for exclusion before filtering
    exclude_boxes = None
    if exclude_phrase is not None and len(boxes) > 0:
        exclude_mask = [exclude_phrase.lower() in phrase.lower() for phrase in phrases]
        if any(exclude_mask):
            exclude_boxes = boxes[exclude_mask]
            print(f"Found {sum(exclude_mask)} boxes to exclude with phrase '{exclude_phrase}'")
    
    # Filter boxes by phrase if specified
    if filter_phrase is not None and len(boxes) > 0:
        # Keep only boxes where phrase contains filter_phrase
        mask = [filter_phrase.lower() in phrase.lower() for phrase in phrases]
        if any(mask):
            boxes = boxes[mask]
            logits = logits[mask]
            phrases = [p for p, m in zip(phrases, mask) if m]
            print(f"Filtered to phrases: {phrases}")
        else:
            print(f"Warning: No phrases contain '{filter_phrase}', using all detections")
    
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB

    # set image
    sam_predictor.set_image(image_source)
    # box: normalized box xywh -> unnormalized xyxy
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    if len(boxes_xyxy) > 0:
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)
        masks, _, _ = sam_predictor.predict_torch(
                    point_coords = None,
                    point_labels = None,
                    boxes = transformed_boxes,
                    multimask_output = False,
                )
    else:
        masks = torch.zeros((1,1,H,W)).cuda()
    
    # Generate mask for excluded regions
    exclude_mask_combined = None
    if exclude_boxes is not None and len(exclude_boxes) > 0:
        exclude_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(exclude_boxes) * torch.Tensor([W, H, W, H])
        transformed_exclude_boxes = sam_predictor.transform.apply_boxes_torch(exclude_boxes_xyxy, image_source.shape[:2]).to(device)
        exclude_masks, _, _ = sam_predictor.predict_torch(
                    point_coords = None,
                    point_labels = None,
                    boxes = transformed_exclude_boxes,
                    multimask_output = False,
                )
        # Combine all exclude masks
        exclude_mask_combined = torch.sum(exclude_masks, dim=0).squeeze().bool()
        print(f"Exclude mask created: {exclude_mask_combined.sum().item()} pixels")

    # Combine masks and remove excluded regions
    final_mask = torch.sum(masks, dim=0).squeeze().bool()
    
    if exclude_mask_combined is not None:
        original_count = final_mask.sum().item()
        final_mask = final_mask & (~exclude_mask_combined)  # Remove excluded regions
        new_count = final_mask.sum().item()
        print(f"Removed {original_count - new_count} pixels from exclude regions")

    for i in range(len(masks)):
        annotated_frame_with_mask = show_mask(masks[i][0].cpu().numpy(), annotated_frame)
    
    return final_mask, annotated_frame_with_mask


def select_obj_ioa(classification_maps, mask, ioa_thresh=0.7):
    unique_classes = classification_maps.unique()
    classes_above_threshold = []

    for class_id in unique_classes:
        class_mask = (classification_maps == class_id).byte()
        class_area = class_mask.sum()
        intersection = torch.sum(class_mask * mask)
        ioa = intersection / class_area if class_area != 0 else 0  # Avoid division by zero
        
        if ioa > ioa_thresh:
            classes_above_threshold.append(class_id)

    return torch.tensor(classes_above_threshold).cuda()


