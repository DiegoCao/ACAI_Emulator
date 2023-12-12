"""
Credit to UMich EECS442
By Churong Ji
Please do NOT copy for personal use
Contact: churongj@andrew.cmu.edu
"""
import matplotlib.pyplot as plt
import numpy as np
# import cv2
import torch

# data type and device for torch.tensor
# unpack as argument to torch functions, like so: **to_float
to_float = {'dtype': torch.float, 'device': 'cpu'}
to_float_cuda = {'dtype': torch.float, 'device': 'cuda'}
to_double = {'dtype': torch.double, 'device': 'cpu'}
to_double_cuda = {'dtype': torch.double, 'device': 'cuda'}
to_long = {'dtype': torch.long, 'device': 'cpu'}
to_long_cuda = {'dtype': torch.long, 'device': 'cuda'}


def coord_trans(bbox, w_pixel, h_pixel, w_amap=7, h_amap=7, mode='a2p'):
    """
    Coordinate transformation function. It converts the box coordinate from
    the image coordinate system to the activation map coordinate system and vice versa.
    In our case, the input image will have a few hundred pixels in
    width/height while the activation map is of size 7x7.

    Input:
    - bbox: Could be either bbox, anchor, or proposal, of shape Bx*x4
    - w_pixel: Number of pixels in the width side of the original image, of shape B
    - h_pixel: Number of pixels in the height side of the original image, of shape B
    - w_amap: Number of pixels in the width side of the activation map, scalar
    - h_amap: Number of pixels in the height side of the activation map, scalar
    - mode: Whether transfer from the original image to activation map ('p2a') or
          the opposite ('a2p')

    Output:
    - resized_bbox: Resized box coordinates, of the same shape as the input bbox
    """

    assert mode in ('p2a', 'a2p'), 'invalid coordinate transformation mode!'
    assert bbox.shape[-1] >= 4, 'the transformation is applied to the first 4 values of dim -1'

    if bbox.shape[0] == 0:  # corner cases
        return bbox

    resized_bbox = bbox.clone()
    # could still work if the first dim of bbox is not batch size
    # in that case, w_pixel and h_pixel will be scalars
    resized_bbox = resized_bbox.view(bbox.shape[0], -1, bbox.shape[-1])
    invalid_bbox_mask = (resized_bbox == -1)  # indicating invalid bbox

    if mode == 'p2a':
        # pixel to activation
        width_ratio = w_pixel * 1. / w_amap
        height_ratio = h_pixel * 1. / h_amap
        resized_bbox[:, :, [0, 2]] /= width_ratio.view(-1, 1, 1)
        resized_bbox[:, :, [1, 3]] /= height_ratio.view(-1, 1, 1)
    else:
        # activation to pixel
        width_ratio = w_pixel * 1. / w_amap
        height_ratio = h_pixel * 1. / h_amap
        resized_bbox[:, :, [0, 2]] *= width_ratio.view(-1, 1, 1)
        resized_bbox[:, :, [1, 3]] *= height_ratio.view(-1, 1, 1)

    resized_bbox.masked_fill_(invalid_bbox_mask, -1)
    resized_bbox.resize_as_(bbox)
    return resized_bbox
