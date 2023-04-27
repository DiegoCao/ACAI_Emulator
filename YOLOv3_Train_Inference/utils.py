import matplotlib.pyplot as plt
import numpy as np
import cv2
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


def data_visualizer(img, idx_to_class, bbox=None, pred=None):
    """
    Data visualizer on the original image. Support both GT box input and proposal 
    input.

    Input:
    - img: PIL Image input
    - idx_to_class: Mapping from the index (0-19) to the class name
    - bbox: GT bbox (in red, optional), a tensor of shape Nx5, where N is
          the number of GT boxes, 5 indicates (x_tl, y_tl, x_br, y_br, class)
    - pred: Predicted bbox (in green, optional), a tensor of shape N'x6, where
          N' is the number of predicted boxes, 6 indicates
          (x_tl, y_tl, x_br, y_br, class, object confidence score)
    """

    img_copy = np.array(img).astype('uint8')

    if bbox is not None:
        for bbox_idx in range(bbox.shape[0]):
            one_bbox = bbox[bbox_idx][:4]
            cv2.rectangle(img_copy, (int(one_bbox[0]), int(one_bbox[1])), (int(one_bbox[2]),
                                                                           int(one_bbox[3])), (255, 0, 0), 2)
            if bbox.shape[1] > 4:  # if class info provided
                obj_cls = idx_to_class[bbox[bbox_idx][4].item()]
                cv2.putText(img_copy, '%s' % obj_cls,
                            (one_bbox[0], one_bbox[1] + 15),
                            cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)

    if pred is not None:
        for bbox_idx in range(pred.shape[0]):
            one_bbox = pred[bbox_idx][:4]
            cv2.rectangle(img_copy, (int(one_bbox[0]), int(one_bbox[1])), (int(one_bbox[2]),
                                                                           int(one_bbox[3])), (0, 255, 0), 2)

            if pred.shape[1] > 4:  # if class and conf score info provided
                obj_cls = idx_to_class[pred[bbox_idx][4].item()]
                conf_score = pred[bbox_idx][5].item()
                cv2.putText(img_copy, '%s, %.2f' % (obj_cls, conf_score),
                            (one_bbox[0], one_bbox[1] + 15),
                            cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)

    plt.imshow(img_copy)
    plt.axis('off')
    plt.show()
