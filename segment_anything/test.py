"""
1、用dataloader导入shot张support images用来微调一次，1张query image用来测试，微调和测试时候的mask都同样处理 ✔
2、处理好模型预测的mask和groud_truth的关系
3、记录下每一个类别的iou，参考其他代码怎么写的 ✔
"""
import os
import random

import numpy as np
import torch
import cv2
import argparse
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor, iou_calculate
from segment_anything.utils import getClasses, testdataset
from util import AverageMeter


def get_model():
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device="cuda")
    optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = torch.nn.BCELoss()
    return sam, optimizer, loss_fn


def get_parser():
    parser = argparse.ArgumentParser(description="Finetuning Segment Anything Model")
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--dataset', type=str, default='Animal')
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--test_num', type=int, default=1000)
    parser.add_argument('--test_epoch', type=int, default=2)
    parser.add_argument('--checkpoint', type=str, default='../checkpoint/sam_vit_b_01ec64.pth')
    parser.add_argument('--model_type', type=str, default='vit_b')
    parser.add_argument('--lr', type=int, default=1e-4)
    parser.add_argument('--wd', type=int, default=0)
    args = parser.parse_args()
    return args


def setup_seed(seed=2023):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# 输入的image形式为(h, w, c)，没有batch这个维度，batch_size只能为1
def finetune(support_imgs, support_labels):
    for i in range(support_imgs[1]):
        image = support_imgs[0, i]
        gt_label = support_labels[0, i]

        sam, optimizer, loss_fn = get_model()
        mask_generator = SamAutomaticMaskGenerator(sam)
        # masks是由sam的everything功能生成的一系列mask
        masks = mask_generator.generate(image)
        _, _, mask = iou_calculate.choose_mask(masks, gt_label)
        loss = loss_fn(mask, gt_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return sam


def test():
    global args, logger
    args = get_parser()
    print(args)

    setup_seed()

    num_classes, class_names = getClasses(args.data_root, args.shot, args.dataset)
    test_dataset = testdataset.TestData(class_num=num_classes, class_names=class_names, test_num=args.test_num, data_root=args.data_root, data_set=args.dataset)
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=False, sampler=None)

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    class_intersection_meter = [0] * num_classes
    class_union_meter = [0] * num_classes

    for e in range(args.test_epoch):
        for i, (support_imgs, support_labels, query_img, query_label, cls) in enumerate(dataloader):
            # batch to cuda
            model = finetune(support_imgs, support_labels)
            with torch.no_grad():
                # 重新输入模型
                mask_generator = SamAutomaticMaskGenerator(model)
                masks = mask_generator.generate(query_img)
                intersection, union, _ = iou_calculate.choose_mask(masks, query_label)

            cls = cls[0].cpu().numpy()[0]
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union)
            class_intersection_meter[cls] += intersection[1]
            class_union_meter[cls] += union[1]

    class_iou_class = []
    class_miou = 0
    for i in range(len(class_intersection_meter)):
        class_iou = class_intersection_meter[i] / (class_union_meter[i] + 1e-10)
        class_iou_class.append(class_iou)
        class_miou += class_iou

    class_miou = class_miou * 1.0 / len(class_intersection_meter)
    logger.info('meanIoU---Val result: mIoU_f {:.4f}.'.format(class_miou))  # final
    for i in range(num_classes):
        logger.info('Class_{} Result: iou_f {:.4f}.'.format(i+1, class_iou_class[i]))
