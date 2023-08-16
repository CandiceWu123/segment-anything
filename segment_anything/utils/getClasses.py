import os


def get_classes(base_dir, n_shots, dataset_name):
    cls_names = []
    cls_num = 0
    if dataset_name == 'Animal' or dataset_name == 'Magnetic_tile_surface' or dataset_name == 'Artificial_Luna_Landscape':
        dataset_path = os.path.join(base_dir, dataset_name)
        for cls_name in os.listdir(dataset_path):
            if len(os.listdir(os.path.join(dataset_path, cls_name, 'support', 'images'))) >= n_shots:
                cls_names.append(cls_name)
    else:
        cls_names = [dataset_name]
    cls_num = len(cls_names)

    return cls_num, cls_names

