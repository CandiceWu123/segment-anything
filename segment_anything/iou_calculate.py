# input: masks
# 寻找与ground truth最接近的mask，也就是miou最大的mask
import torch

# choose the mask with highest iou
def choose_mask(masks, gt_label):
    max_iou = -1
    for i in range(len(masks)):
        mask = masks[i]['segmentation']
        # transform mask to float type
        mask = torch.tensor(mask)
        target_idx = torch.where(mask == True)
        mask = torch.zeros(mask.shape)
        mask[target_idx] = 1.0

        iou, temp_intersection, temp_union = calculate_iou(mask, gt_label)
        # print(iou)
        if iou[1] > max_iou:
            final_mask = mask
            intersection = temp_intersection
            union = temp_union
    return intersection, union, final_mask


def calculate_iou(mask, gt_label):
    assert mask.shape == gt_label.shape
    mask = mask.reshape(-1)
    gt_label = gt_label.reshape(-1)
    intersection = mask[mask == gt_label]
    area_intersection = torch.histc(intersection, bins=2, min=0, max=1)
    area_mask = torch.histc(mask, bins=2, min=0, max=1)
    area_target = torch.histc(gt_label, bins=2, min=0, max=1)
    area_union = area_mask + area_target - area_intersection
    iou = area_intersection / (area_union + 1e-10)
    return iou, area_intersection, area_union
