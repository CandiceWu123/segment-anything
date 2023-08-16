import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class TestData(Dataset):
    def __init__(self, class_num, class_names, test_num, shot=1, data_root=None, data_set=None):
        self.num_classes = class_num
        self.shot = shot
        self.base_dir = data_root
        self.dataset_name = data_set
        self.max_iters = test_num
        self.cls_names = class_names
        self.ids = []

        temp_ids = []
        cls_ids = []

        # base_dir = C:\Users\86189\Desktop\数据集3.0
        # 直接为support和query上一级的目录
        if self.dataset_name == 'Animal' or self.dataset_name == 'Magnetic_tile_surface' or self.dataset_name == 'Artificial_Luna_Landscape':
            self.path_dir = os.path.join(self.base_dir, self.dataset_name)
        else:
            # self.cls_names = [self.dataset_name]
            self.path_dir = self.base_dir

        for name in self.cls_names:
            for id in os.listdir(os.path.join(self.path_dir, name, 'support', 'images')):
                temp_ids.append(id)
            cls_ids.append(temp_ids)
            temp_ids = []
            for id in os.listdir(os.path.join(self.path_dir, name, 'query', 'images')):
                temp_ids.append(id)
            cls_ids.append(temp_ids)
            self.ids.append(cls_ids)
            cls_ids = []
            temp_ids = []

    def __len__(self):
        return self.max_iters

    def sample_episode(self, idx):
        # 本次采样的类别
        # print("idx", idx)
        class_id = idx % len(self.cls_names)
        print("class_id", class_id)
        # support_names = np.random.choice(self.ids[class_id][0], self.shot, replace=False)
        query_name = np.random.choice(self.ids[class_id][1], 1, replace=False)
        return query_name[0], class_id

    def process_label(self, label):
        if label.mode != 'L':
            label = label.convert('L')
        label = np.array(label)
        label = torch.tensor(label).float()
        target_idx = torch.where(label == 255)
        label = torch.zeros(label.shape)
        label[target_idx] = 1
        return label

    def __getitem__(self, idx):  # It only gives the query
        query_name, class_id = self.sample_episode(idx)
        print(query_name)
        ### Read and process query image + mask
        image_path = os.path.join(self.path_dir, self.cls_names[class_id], 'query', 'images', query_name)
        label_path = os.path.join(self.path_dir, self.cls_names[class_id], 'query', 'masks',
                                  query_name.split('.')[0] + ".png")
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = Image.open(label_path)
        label = self.process_label(label)

        # 根据本次的query图像的类别采样相应的support image
        support_names = np.random.choice(self.ids[class_id][0], self.shot, replace=False)
        support_image_path_list = []
        support_label_path_list = []

        for k in range(self.shot):
            support_image_path = os.path.join(self.path_dir, self.cls_names[class_id], 'support', 'images',
                                              support_names[k])
            support_label_path = os.path.join(self.path_dir, self.cls_names[class_id], 'support', 'masks',
                                              support_names[k].split('.')[0] + ".png")
            support_image_path_list.append(support_image_path)
            support_label_path_list.append(support_label_path)

        support_image_list = []
        support_label_list = []

        for k in range(self.shot):
            support_image_path = support_image_path_list[k]
            support_label_path = support_label_path_list[k]

            support_image = cv2.imread(support_image_path, cv2.IMREAD_COLOR)
            support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)

            support_label = Image.open(support_label_path)
            support_label = self.process_label(support_label)
            support_image_list.append(support_image)
            support_label_list.append(support_label)
        assert len(support_label_list) == self.shot and len(support_image_list) == self.shot

        s_xs = support_image_list
        s_ys = support_label_list
        s_x = s_xs[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_x = torch.cat([s_xs[i].unsqueeze(0), s_x], 0)
        s_y = s_ys[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_y = torch.cat([s_ys[i].unsqueeze(0), s_y], 0)

        # Return
        return image, label, s_x, s_y, class_id

