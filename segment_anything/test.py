"""
1、用dataloader导入shot张support images用来微调一次，1张query image用来测试，微调和测试时候的mask都同样处理 ✔
2、处理好模型预测的mask和groud_truth的关系
3、记录下每一个类别的iou，参考其他代码怎么写的 ✔
"""
import os
import random
import logging
import numpy as np
import torch
import cv2
import argparse
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor, iou_calculate
from segment_anything.utils import getClasses, testdataset
from util import AverageMeter
import datetime
from tqdm import tqdm
import torch.nn.functional as F

def get_model():
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device="cuda")
    optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = torch.nn.CrossEntropyLoss()
    return sam, optimizer, loss_fn


def get_parser():
    parser = argparse.ArgumentParser(description="Finetuning Segment Anything Model")
    parser.add_argument('--data_root', type=str, default=r'D:\Dataset')
    parser.add_argument('--dataset', type=str, default='Animal')
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--test_num', type=int, default=5)
    parser.add_argument('--test_epoch', type=int, default=1)
    parser.add_argument('--checkpoint', type=str, default='../checkpoint/sam_vit_b_01ec64.pth')
    parser.add_argument('--model_type', type=str, default='vit_b')
    parser.add_argument('--lr', type=int, default=1e-4)
    parser.add_argument('--wd', type=int, default=0)
    parser.add_argument('--log_path', type=str, default='./logs')
    args = parser.parse_args()
    return args

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
    log_path = os.path.join(args.log_path, str(args.shot)+'shot')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path, args.dataset + logtime + '.txt')
    file_handler = logging.FileHandler(log_name)
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    file_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.addHandler(file_handler)
    return logger


def setup_seed(seed=2023):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_image_label(image_path, label_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    label = Image.open(label_path)
    if label.mode != 'L':
        label = label.convert('L')
    label = np.array(label)
    label = torch.tensor(label).float()
    target_idx = torch.where(label == 255)
    label = torch.zeros(label.shape)
    label[target_idx] = 1
    return image, label

# 输入的image形式为(h, w, c)，没有batch这个维度，batch_size只能为1
def finetune(support_imgs, support_labels):
    for i in range(len(support_imgs)):
        image_path = support_imgs[i][0]
        gt_label_path = support_labels[i][0]
        image, gt_label = load_image_label(image_path, gt_label_path)

        print("image", image.shape)
        print("image type", type(image))
        sam, optimizer, loss_fn = get_model()
        mask_generator = SamAutomaticMaskGenerator(sam)
        # masks是由sam的everything功能生成的一系列mask
        masks = mask_generator.generate(image)
        print("label", gt_label.shape)
        print("label type", type(gt_label))
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

    logger = get_logger()

    setup_seed()

    num_classes, class_names = getClasses.get_classes(args.data_root, args.shot, args.dataset)
    test_dataset = testdataset.TestData(class_num=num_classes, class_names=class_names, test_num=args.test_num, shot=args.shot, data_root=args.data_root, data_set=args.dataset)
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=False, sampler=None)

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    class_intersection_meter = [0] * num_classes
    class_union_meter = [0] * num_classes

    for e in range(args.test_epoch):
        for i, (query_img_path, query_label_path, support_imgs_path, support_labels_path, cls) in enumerate(tqdm(dataloader)):
            model = finetune(support_imgs_path, support_labels_path)
            with torch.no_grad():
                # 重新输入模型
                mask_generator = SamAutomaticMaskGenerator(model)
                query_img_path = query_img_path[0]
                query_label_path = query_label_path[0]
                query_img, query_label = load_image_label(query_img_path, query_label_path)
                print("query image", query_img.shape)
                print("query image type", type(query_img))
                print("query label", query_label.shape)
                print("image type", type(query_label))
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
    logger.info('meanIoU---Val result: mIoU {:.4f}.'.format(class_miou))  # final
    for i in range(num_classes):
        logger.info('Class_{} Result: iou {:.4f}.'.format(i+1, class_iou_class[i]))

if __name__ == '__main__':
    test()