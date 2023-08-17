import os
import random
import logging
import numpy as np
import torch
import cv2
import argparse
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor, iou_calculate
from segment_anything.test import load_image_label
from segment_anything.utils.transforms import ResizeLongestSide
import iou_calculate
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor, iou_calculate
from torch.nn.functional import  normalize, threshold

sam = sam_model_registry['vit_b'](checkpoint=r'D:\segment-anything\checkpoint\sam_vit_b_01ec64.pth')
sam.to(device="cuda")

points_per_side = 16
points_per_batch = 64
image_path = r'D:\Dataset\Animal\butterfly\support\images\butterfly-339934_640_30.jpg'
label_path = r'D:\Dataset\Animal\butterfly\support\masks\butterfly-339934_640_30.png'

def build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points

def batch_iterator(batch_size: int, *args):
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]


def finetune():
    # model

    optimize_param = []
    for parameter in sam.parameters():
        parameter.requires_grad = False
    for name, parameter in sam.mask_decoder.named_parameters():
        parameter.requires_grad = True
        optimize_param.append(parameter)
    optimizer = torch.optim.Adam(optimize_param, lr=1e-4, weight_decay=0.0)
    loss_fn = torch.nn.BCEWithLogitsLoss()


    # process image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_size = image.shape[:2]
    transform = ResizeLongestSide(sam.image_encoder.img_size)
    input_image = transform.apply_image(image)
    input_image_torch = torch.as_tensor(input_image, device='cuda')

    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    input_size = tuple(input_image_torch.shape[-2:])
    input_image = sam.preprocess(input_image_torch)

    # process gt_label
    label = Image.open(label_path)
    if label.mode != 'L':
        label = label.convert('L')
    label = np.array(label)
    label = torch.tensor(label).float()
    target_idx = torch.where(label == 255)
    label = torch.zeros(label.shape)
    label[target_idx] = 1
    label = label.cuda()

    # 生成prompt points
    points_orig = build_point_grid(points_per_side)
    points_scale = np.array(orig_size)[None, ::-1]
    points_for_image = points_orig * points_scale

    i=0
    for (points,) in batch_iterator(points_per_batch, points_for_image):
        print("batch", i)
        i= i+1
        orig_h, orig_w = orig_size
        transformed_points = transform.apply_coords(points, orig_size)
        in_points = torch.as_tensor(transformed_points, device='cuda')
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device='cuda')

        point_coords = in_points[:, None, :]
        point_labels = in_labels[:, None]
        points = (point_coords, point_labels)
        with torch.no_grad():
            features = sam.image_encoder(input_image)

            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                        points=points,
                        boxes=None,
                        masks=None
                    )
        low_res_masks, iou_predictions = sam.mask_decoder(
                image_embeddings=features,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )
        # print(low_res_masks.requires_grad)
        masks = sam.postprocess_masks(low_res_masks, input_size, orig_size).to('cuda')
        masks = threshold(masks, 0.0, 0)
        one_idx = torch.where(masks > 0)
        masks[one_idx] = 1

        mask = iou_calculate.choose_mask_finetune(masks, label)
        print(mask.requires_grad)
        optimizer.zero_grad()
        loss = loss_fn(mask,label)
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    finetune()





