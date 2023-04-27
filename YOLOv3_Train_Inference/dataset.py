import torch
import random
from torchvision import transforms

class_to_idx = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
                'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10,
                'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15,
                'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19
                }
idx_to_class = {i: c for c, i in class_to_idx.items()}


def get_pascal_voc2007_data(image_root, split='train'):
    """
    Use torchvision.datasets
    https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.VOCDetection
    """
    from torchvision import datasets

    train_dataset = datasets.VOCDetection(image_root, year='2007', image_set=split,
                                          download=True)

    return train_dataset


def pascal_voc2007_loader(dataset, batch_size, num_workers=0):
    """
    Data loader for Pascal VOC 2007.
    https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """
    from torch.utils.data import DataLoader
    # turn off shuffle so we can index the original image
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False, pin_memory=True,
                        collate_fn=voc_collate_fn)
    return loader


def voc_collate_fn(batch_lst, reshape_size=224):
    preprocess = transforms.Compose([
        transforms.Resize((reshape_size, reshape_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    batch_size = len(batch_lst)

    img_batch = torch.zeros(batch_size, 3, reshape_size, reshape_size)

    max_num_box = max(len(batch_lst[i][1]['annotation']['object']) \
                      for i in range(batch_size))

    box_batch = torch.Tensor(batch_size, max_num_box, 5).fill_(-1.)
    w_list = []
    h_list = []
    img_id_list = []

    for i in range(batch_size):
        img, ann = batch_lst[i]
        w_list.append(img.size[0])  # image width
        h_list.append(img.size[1])  # image height
        img_id_list.append(ann['annotation']['filename'])
        img_batch[i] = preprocess(img)
        all_bbox = ann['annotation']['object']
        if type(all_bbox) == dict:  # inconsistency in the annotation file
            all_bbox = [all_bbox]

        for bbox_idx, one_bbox in enumerate(all_bbox):
            bbox = one_bbox['bndbox']
            obj_cls = one_bbox['name']
            box_batch[i][bbox_idx] = torch.Tensor([float(bbox['xmin']), float(bbox['ymin']),
                                                   float(bbox['xmax']), float(bbox['ymax']), class_to_idx[obj_cls]])

    h_batch = torch.tensor(h_list)
    w_batch = torch.tensor(w_list)

    return img_batch, box_batch, w_batch, h_batch, img_id_list


def filter_dataset_with_class(dataset, class_name, ratio):
    images_with_class = []
    images_without_class = []
    final_images = []
    for img, ann in dataset:
        objects = ann['annotation']['object']
        if type(objects) == dict:
            objects = [objects]
        names = [obj['name'] for obj in objects]
        if class_name in names:
            images_with_class.append((img, ann))
        else:
            images_without_class.append((img, ann))
    target_num = len(images_with_class)
    non_target_num = len(images_without_class)
    cur_ratio = target_num / (target_num + non_target_num)
    if ratio > cur_ratio:
        final_images.extend(images_with_class)
        final_images.extend(random.sample(images_without_class, int((1 - ratio) / ratio * target_num)))
    else:
        final_images.extend(images_without_class)
        final_images.extend(random.sample(images_with_class, int(ratio / (1 - ratio) * non_target_num)))
    random.shuffle(final_images)
    return final_images
