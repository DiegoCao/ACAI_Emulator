"""
Credit to UMich EECS442
By Churong Ji
Please do NOT copy for personal use
Contact: churongj@andrew.cmu.edu
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np


class FeatureExtractor(nn.Module):
    """
    Image feature extraction with MobileNet.
    """

    def __init__(self, reshape_size=224, pooling=False, verbose=False):
        super().__init__()

        from torchvision import models
        from torchsummary import summary

        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.mobilenet = nn.Sequential(*list(self.mobilenet.children())[:-1])  # Remove the last classifier

        # average pooling
        if pooling:
            self.mobilenet.add_module('LastAvgPool',
                                      nn.AvgPool2d(math.ceil(reshape_size / 32.)))  # input: N x 1280 x 7 x 7

        for i in self.mobilenet.named_parameters():
            i[1].requires_grad = True  # fine-tune all parameters

        if verbose:
            summary(self.mobilenet.cuda(), (3, reshape_size, reshape_size))

    def forward(self, img, verbose=False):
        """
        Inputs:
        - img: Batch of resized images, of shape Nx3x224x224

        Outputs:
        - feat: Image feature, of shape Nx1280 (pooled) or Nx1280x7x7
        """
        num_img = img.shape[0]

        img_prepro = img

        feat = []
        process_batch = 500
        for b in range(math.ceil(num_img / process_batch)):
            feat.append(self.mobilenet(img_prepro[b * process_batch:(b + 1) * process_batch]
                                       ).squeeze(-1).squeeze(-1))  # forward and squeeze
        feat = torch.cat(feat)

        if verbose:
            print('Output feature shape: ', feat.shape)

        return feat


def GenerateGrid(batch_size, w_amap=7, h_amap=7, dtype=torch.float32, device='cpu'):
    """
    Return a grid cell given a batch size (center coordinates).

    Inputs:
    - batch_size, B
    - w_amap: or W', width of the activation map (number of grids in the horizontal dimension)
    - h_amap: or H', height of the activation map (number of grids in the vertical dimension)
    - W' and H' are always 7 in our case while w and h might vary.

    Outputs:
    grid: A float32 tensor of shape (B, H', W', 2) giving the (x, y) coordinates
        of the centers of each feature for a feature map of shape (B, D, H', W')
    """
    w_range = torch.arange(0, w_amap, dtype=dtype, device=device) + 0.5
    h_range = torch.arange(0, h_amap, dtype=dtype, device=device) + 0.5

    w_grid_idx = w_range.unsqueeze(0).repeat(h_amap, 1)
    h_grid_idx = h_range.unsqueeze(1).repeat(1, w_amap)
    grid = torch.stack([w_grid_idx, h_grid_idx], dim=-1)
    grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    return grid


def GenerateProposal(grids, offsets):
    """
    Proposal generator.

    Inputs:
    - grids: Activation grids, of shape (B, H', W', 2). Grid centers are 
    represented by their coordinates in the activation map.
    - offsets: Transformations obtained from the Prediction Network 
    of shape (B, A, H', W', 4) that will be used to generate proposals region 
    proposals. The transformation offsets[b, a, h, w] = (tx, ty, tw, th) will be 
    applied to the grids[b, h, w]. 
    Assume that tx and ty are in the range
    (-0.5, 0.5) and h,w are normalized and thus in the range (0, 1).

    Outputs:
    - proposals: Region proposals of shape (B, A, H', W', 4), represented by the
    coordinates of their top-left and bottom-right corners. Using the
    transform offsets[b, a, h, w] and girds[b, a, h, w] should give the proposals.
    The expected parametrization of the proposals is (xtl, ytl, xbr, ybr). 

    CAUTION: 
    Notice that the offsets here are parametrized as (x, y, w, h). 
    The proposals you return need to be of the form (xtl, ytl, xbr, ybr).
    """
    # 1. Follow the formulas above to convert the grid centers into proposals.
    temp = grids.clone()
    new_grids = temp[:, None, :, :, :]
    temp_proposals = offsets.clone()
    proposals = offsets.clone()
    temp_proposals[:, :, :, :, 0] = new_grids[:, :, :, :, 0] + offsets[:, :, :, :, 0]
    temp_proposals[:, :, :, :, 1] = new_grids[:, :, :, :, 1] + offsets[:, :, :, :, 1]
    temp_proposals[:, :, :, :, 2] = offsets[:, :, :, :, 2] * 7
    temp_proposals[:, :, :, :, 3] = offsets[:, :, :, :, 3] * 7

    # 2. Convert the proposals into (xtl, ytl, xbr, ybr) coordinate format as
    # mentioned in the header and in the cell above that.
    proposals[:, :, :, :, 0] = temp_proposals[:, :, :, :, 0] - temp_proposals[:, :, :, :, 2] / 2
    proposals[:, :, :, :, 1] = temp_proposals[:, :, :, :, 1] - temp_proposals[:, :, :, :, 3] / 2
    proposals[:, :, :, :, 2] = temp_proposals[:, :, :, :, 0] + temp_proposals[:, :, :, :, 2] / 2
    proposals[:, :, :, :, 3] = temp_proposals[:, :, :, :, 1] + temp_proposals[:, :, :, :, 3] / 2

    return proposals


def IoU(proposals, bboxes):
    """
    Compute intersection over union between sets of bounding boxes.

    Inputs:
    - proposals: Proposals of shape (B, A, H', W', 4). These should be parametrized
    as (xtl, ytl, xbr, ybr).
    - bboxes: Ground-truth boxes from the DataLoader of shape (B, N, 5).
    Each ground-truth box is represented as tuple (x_tl, y_tl, x_br, y_br, class).
    If image i has fewer than N boxes, then bboxes[i] will be padded with extra
    rows of -1.

    Outputs:
    - iou_mat: IoU matrix of shape (B, A*H'*W', N) where iou_mat[b, i, n] gives
    the IoU between one element of proposals[b] and bboxes[b, n].

    For this implementation you DO NOT need to filter invalid proposals or boxes;
    in particular you don't need any special handling for bboxxes that are padded
    with -1.
    """

    B, A, H, W, _ = proposals.shape
    N = bboxes.shape[1]
    proposals = proposals.reshape(B, A * H * W, 4).clone()
    proposals = torch.repeat_interleave(proposals[:, :, np.newaxis, :], N, 2)
    bboxes = bboxes.clone()
    bbox_prop = bboxes.unsqueeze(1)

    xtl = torch.max(proposals[:, :, :, 0], bbox_prop[:, :, :, 0])
    ytl = torch.max(proposals[:, :, :, 1], bbox_prop[:, :, :, 1])
    xbr = torch.min(proposals[:, :, :, 2], bbox_prop[:, :, :, 2])
    ybr = torch.min(proposals[:, :, :, 3], bbox_prop[:, :, :, 3])

    intersection = torch.clamp(xbr - xtl, min=0) * torch.clamp(ybr - ytl, min=0)
    proposal_area = (proposals[:, :, :, 2] - proposals[:, :, :, 0]) * (proposals[:, :, :, 3] - proposals[:, :, :, 1])
    bbox_area = (bbox_prop[:, :, :, 2] - bbox_prop[:, :, :, 0]) * (bbox_prop[:, :, :, 3] - bbox_prop[:, :, :, 1])

    iou_mat = intersection / (proposal_area + bbox_area - intersection)

    return iou_mat


class PredictionNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_bboxes=2, num_classes=20, drop_ratio=0.3):
        super().__init__()

        assert (num_classes != 0 and num_bboxes != 0)
        self.num_classes = num_classes
        self.num_bboxes = num_bboxes

        # Here we set up a network that will predict outputs for all bounding boxes.
        # This network has a 1x1 convolution layer with `hidden_dim` filters, 
        # followed by a Dropout layer with `p=drop_ratio`, a Leaky ReLU 
        # nonlinearity, and finally another 1x1 convolution layer to predict all
        # outputs. The network is stored in `self.net`, and has shape 
        # (B, 5*A+C, 7, 7), where the 5 predictions are in the order 
        # (x, y, w, h, conf_score), with A = `self.num_bboxes`
        # and C = `self.num_classes`.                                 

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.drop_ratio = drop_ratio
        out_dim = 5 * self.num_bboxes + self.num_classes

        layers = [
            torch.nn.Conv2d(self.in_dim, self.hidden_dim, kernel_size=1),
            torch.nn.Dropout(p=self.drop_ratio),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Conv2d(self.hidden_dim, out_dim, kernel_size=1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, features):
        """
        Run the forward pass of the network to predict outputs given features
        from the backbone network.

        Inputs:
        - features: Tensor of shape (B, in_dim, 7, 7) giving image features computed
          by the backbone network.

        Outputs:
        - bbox_xywh: Tensor of shape (B, A, 4, H, W) giving predicted offsets for 
          all bounding boxes.
        - conf_scores: Tensor of shape (B, A, H, W) giving predicted classification
          scores for all bounding boxes.
        - cls_scores: Tensor of shape (B, C, H, W) giving classification scores for
          each spatial position.
        """
        bbox_xywh, conf_scores, cls_scores = None, None, None

        predictions = self.net(features)

        offsets_conf = predictions[:, :5 * self.num_bboxes, :, :]
        B, _, H, W = offsets_conf.shape
        offsets_conf = offsets_conf.reshape(B, self.num_bboxes, 5, H, W)

        bbox_xywh = torch.zeros_like(offsets_conf[:, :, 0:4, :, :])
        bbox_xywh[:, :, 0:2, :, :] = torch.sigmoid(offsets_conf[:, :, 0:2, :, :].clone()) - 0.5
        bbox_xywh[:, :, 2:4, :, :] = torch.sigmoid(offsets_conf[:, :, 2:4, :, :].clone())

        conf_scores = torch.sigmoid(offsets_conf[:, :, 4, :, :].clone())

        cls_scores = predictions[:, 5 * self.num_bboxes:, :, :].clone()

        return bbox_xywh, conf_scores, cls_scores


def ReferenceOnActivatedBboxes(bboxes, gt_bboxes, grid, iou_mat, pos_thresh=0.7, neg_thresh=0.3):
    """
    Determine the activated (positive) and negative bboxes for model training.

    A grid cell is responsible for predicting a GT box if the center of
    the box falls into that cell.
    Implementation details: First compute manhattan distance between grid cell centers
    (BxH’xW’) and GT box centers (BxN). This gives us a matrix of shape Bx(H'xW')xN and
    perform torch.min(dim=1)[1] on it gives us the indexes indicating activated grids
    responsible for GT boxes (convert to x and y). Among all the bboxes associated with
    the activate grids, the bbox with the largest IoU with the GT box is responsible to
    predict (regress to) the GT box.
    Note: One bbox might match multiple GT boxes.

    Main steps include:
    i) Decide activated and negative bboxes based on the IoU matrix.
    ii) Compute GT confidence score/offsets/object class on the positive proposals.
    iii) Compute GT confidence score on the negative proposals.

    Inputs:
    - bboxes: Bounding boxes, of shape BxAxH’xW’x4
    - gt_bboxes: GT boxes of shape BxNx5, where N is the number of PADDED GT boxes,
            5 indicates (x_{lr}^{gt}, y_{lr}^{gt}, x_{rb}^{gt}, y_{rb}^{gt}) and class index
    - grid (float): A cell grid of shape BxH'xW'x2 where 2 indicate the (x, y) coord
    - iou_mat: IoU matrix of shape Bx(AxH’xW’)xN
    - pos_thresh: Positive threshold value
    - neg_thresh: Negative threshold value

    Outputs:
    - activated_anc_ind: Index on activated bboxes, of shape M, where M indicates the 
                       number of activated bboxes
    - negative_anc_ind: Index on negative bboxes, of shape M
    - GT_conf_scores: GT IoU confidence scores on activated bboxes, of shape M
    - GT_offsets: GT offsets on activated bboxes, of shape Mx4. They are denoted as
                \hat{t^x}, \hat{t^y}, \hat{t^w}, \hat{t^h} in the formulation earlier.
    - GT_class: GT class category on activated bboxes, essentially indexed from gt_bboxes[:, :, 4],
              of shape M
    - activated_anc_coord: Coordinates on activated bboxes (mainly for visualization purposes)
    - negative_anc_coord: Coordinates on negative bboxes (mainly for visualization purposes)
    """

    B, A, h_amap, w_amap, _ = bboxes.shape
    N = gt_bboxes.shape[1]

    # activated/positive bboxes
    max_iou_per_anc, max_iou_per_anc_ind = iou_mat.max(dim=-1)

    bbox_mask = (gt_bboxes[:, :, 0] != -1)  # BxN, indicate invalid boxes
    bbox_centers = (gt_bboxes[:, :, 2:4] - gt_bboxes[:, :, :2]) / 2. + gt_bboxes[:, :, :2]  # BxNx2

    mah_dist = torch.abs(grid.view(B, -1, 2).unsqueeze(2) - bbox_centers.unsqueeze(1)).sum(dim=-1)  # Bx(H'xW')xN
    min_mah_dist = mah_dist.min(dim=1, keepdim=True)[0]  # Bx1xN
    grid_mask = (mah_dist == min_mah_dist).unsqueeze(1)  # Bx1x(H'xW')xN

    reshaped_iou_mat = iou_mat.view(B, A, -1, N)
    anc_with_largest_iou = reshaped_iou_mat.max(dim=1, keepdim=True)[0]  # Bx1x(H’xW’)xN
    anc_mask = (anc_with_largest_iou == reshaped_iou_mat)  # BxAx(H’xW’)xN
    activated_anc_mask = (grid_mask & anc_mask).view(B, -1, N)
    activated_anc_mask &= bbox_mask.unsqueeze(1)

    # one bbox could match multiple GT boxes
    activated_anc_ind = torch.nonzero(activated_anc_mask.view(-1)).squeeze(-1)
    GT_conf_scores = iou_mat.view(-1)[activated_anc_ind]
    gt_bboxes = gt_bboxes.view(B, 1, N, 5).repeat(1, A * h_amap * w_amap, 1, 1).view(-1, 5)[activated_anc_ind]
    GT_class = gt_bboxes[:, 4].long()
    gt_bboxes = gt_bboxes[:, :4]
    activated_anc_ind = (activated_anc_ind.float() / activated_anc_mask.shape[-1]).long()

    # print('number of pos proposals: ', activated_anc_ind.shape[0])

    activated_anc_coord = bboxes.reshape(-1, 4)[activated_anc_ind]

    activated_grid_coord = grid.repeat(1, A, 1, 1, 1).reshape(-1, 2)[activated_anc_ind]

    # GT offsets

    # bbox are x_tl, y_tl, x_br, y_br
    # offsets are t_x, t_y, t_w, t_h

    # Grid: (B, H, W, 2) -> This will be used to calculate center offsets
    # w, h offsets are not offsets but normalized w,h themselves.

    wh_offsets = torch.sqrt((gt_bboxes[:, 2:4] - gt_bboxes[:, :2]) / 7.)
    assert torch.max(
        (gt_bboxes[:, 2:4] - gt_bboxes[:, :2]) / 7.) <= 1, "w and h targets not normalised, should be between 0 and 1"

    xy_offsets = (gt_bboxes[:, :2] + gt_bboxes[:, 2:4]) / 2. - activated_grid_coord

    assert torch.max(torch.abs(xy_offsets)) <= 0.5, \
        "x and y offsets should be between -0.5 and 0.5! Got {}".format(
            torch.max(torch.abs(xy_offsets)))

    GT_offsets = torch.cat((xy_offsets, wh_offsets), dim=-1)

    # negative bboxes
    negative_anc_mask = (max_iou_per_anc < neg_thresh)  # Bx(AxH’xW’)
    negative_anc_ind = torch.nonzero(negative_anc_mask.view(-1)).squeeze(-1)
    negative_anc_ind = negative_anc_ind[torch.randint(0, negative_anc_ind.shape[0], (activated_anc_ind.shape[0],))]
    negative_anc_coord = bboxes.reshape(-1, 4)[negative_anc_ind.view(-1)]

    # activated_anc_coord and negative_anc_coord are mainly for visualization purposes
    return activated_anc_ind, negative_anc_ind, GT_conf_scores, GT_offsets, GT_class, \
        activated_anc_coord, negative_anc_coord


def ConfScoreRegression(conf_scores, GT_conf_scores):
    """
    Use sum-squared error as in YOLO

    Inputs:
    - conf_scores: Predicted confidence scores
    - GT_conf_scores: GT confidence scores

    Outputs:
    - conf_score_loss
    """
    # the target conf_scores for negative samples are zeros
    GT_conf_scores = torch.cat((torch.ones_like(GT_conf_scores),
                                torch.zeros_like(GT_conf_scores)), dim=0).view(-1, 1)

    conf_score_loss = torch.sum((conf_scores - GT_conf_scores) ** 2) * 1. / GT_conf_scores.shape[0]
    return conf_score_loss


def BboxRegression(offsets, GT_offsets):
    """"
    Use sum-squared error as in YOLO
    For both xy and wh.
    NOTE: In YOLOv1, the authors use sqrt(w) and sqrt(h) for normalized w and h
    (read paper for more details) and thus both offsets and GT_offsets will 
    be having (x, y, sqrt(w), sqrt(h)) parametrization of the coodinates. 


    Inputs:
    - offsets: Predicted box offsets
    - GT_offsets: GT box offsets

    Outputs:
    - bbox_reg_loss
    """

    bbox_reg_loss = torch.sum((offsets - GT_offsets) ** 2) * 1. / GT_offsets.shape[0]
    return bbox_reg_loss


def ObjectClassification(class_prob, GT_class, batch_size, anc_per_img, activated_anc_ind):
    """"
    Use softmax loss

    Inputs:
    - class_prob: Predicted class logits
    - GT_class: GT box class label
    - batch_size: the batch size to compute loss over
    - anc_per_img: anchor indices for each image
    - activated_anc_ind: indices for positive anchors

    Outputs:
    - object_cls_loss, the classification loss for object detection
    """
    # average within sample and then average across batch
    # such that the class pred would not bias towards dense popular objects like `person`

    all_loss = F.cross_entropy(class_prob, GT_class, reduction='none')  # , reduction='sum') * 1. / batch_size
    object_cls_loss = 0
    for idx in range(batch_size):
        anc_ind_in_img = (activated_anc_ind >= idx * anc_per_img) & (activated_anc_ind < (idx + 1) * anc_per_img)
        object_cls_loss += all_loss[anc_ind_in_img].sum() * 1. / torch.sum(anc_ind_in_img)
    object_cls_loss /= batch_size
    # object_cls_loss = F.cross_entropy(class_prob, GT_class, reduction='sum') * 1. / batch_size

    return object_cls_loss


class SingleStageDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.feat_extractor = FeatureExtractor()
        self.num_classes = 20
        self.num_bboxes = 2
        self.pred_network = PredictionNetwork(1280, num_bboxes=2,
                                              num_classes=self.num_classes)

    def forward(self, images, bboxes):
        """
        Training-time forward pass for the single-stage detector.

        Inputs:
        - images: Input images, of shape (B, 3, 224, 224)
        - bboxes: GT bounding boxes of shape (B, N, 5) (padded)

        Outputs:
        - total_loss: Torch scalar giving the total loss for the batch.
        """
        # weights to multiple to each loss term
        w_conf = 1  # for conf_scores
        w_reg = 1  # for offsets
        w_cls = 1  # for class_prob

        total_loss = None

        # 1. Feature extraction
        features = self.feat_extractor(images)

        # 2. Grid generator
        grid_list = GenerateGrid(images.shape[0])

        # 3. Prediction Network
        bbox_xywh, conf_scores, cls_scores = self.pred_network(features)
        # (B, A, 4, H, W), (B, A, H, W), (B, C, H, W)

        B, A, _, H, W = bbox_xywh.shape
        bbox_xywh = bbox_xywh.permute(0, 1, 3, 4, 2)  # (B, A, H, W, 4)

        assert bbox_xywh.max() <= 1 and bbox_xywh.min() >= -0.5, 'invalid offsets values'

        # 4. Calculate the proposals
        proposals = GenerateProposal(grid_list, bbox_xywh)

        # 5. Compute IoU
        iou_mat = IoU(proposals, bboxes)

        # 7. Using the activated_anc_ind, select the activated conf_scores, bbox_xywh, cls_scores
        activated_anc_ind, negative_anc_ind, GT_conf_scores, GT_offsets, GT_class, _, _ \
            = ReferenceOnActivatedBboxes(bbox_xywh, bboxes, grid_list, iou_mat, neg_thresh=0.3)

        conf_scores = conf_scores.view(B, A, 1, H, W)
        pos = self._extract_bbox_data(conf_scores, activated_anc_ind)
        neg = self._extract_bbox_data(conf_scores, negative_anc_ind)
        conf_scores = torch.cat([pos, neg], dim=0)

        # 6. The loss function
        bbox_xywh[:, :, :, :, 2:4] = torch.sqrt(bbox_xywh[:, :, :, :, 2:4])

        # assert bbox_xywh[:, :, :, :, :2].max() <= 0.5
        # and bbox_xywh[:, :, :, :, :2].min() >= -0.5, 'invalid offsets values'
        # assert bbox_xywh[:, :, :, :, :2:4].max() <= 1
        # and bbox_xywh[:, :, :, :, 2:4].min() >= 0, 'invalid offsets values'

        offsets = self._extract_bbox_data(bbox_xywh.permute(0, 1, 4, 2, 3), activated_anc_ind)
        cls_scores = self._extract_class_scores(cls_scores, activated_anc_ind)
        anc_per_img = torch.prod(torch.tensor(bbox_xywh.shape[1:-1]))  # use as argument in ObjectClassification

        # 8. Compute losses
        conf_loss = ConfScoreRegression(conf_scores, GT_conf_scores)
        reg_loss = BboxRegression(offsets, GT_offsets)
        cls_loss = ObjectClassification(cls_scores, GT_class, B, anc_per_img, activated_anc_ind)

        total_loss = w_conf * conf_loss + w_reg * reg_loss + w_cls * cls_loss

        print('(weighted) conf loss: {:.4f}, reg loss: {:.4f}, cls loss: {:.4f}'.format(conf_loss, reg_loss, cls_loss))

        return total_loss

    def inference(self, images, thresh=0.5, nms_thresh=0.7):
        """"
        Inference-time forward pass for the single stage detector.

        Inputs:
        - images: Input images
        - thresh: Threshold value on confidence scores
        - nms_thresh: Threshold value on NMS

        Outputs:
        - final_propsals: Keeped proposals after confidence score thresholding and NMS,
                        a list of B (*x4) tensors
        - final_conf_scores: Corresponding confidence scores, a list of B (*x1) tensors
        - final_class: Corresponding class predictions, a list of B  (*x1) tensors
        """
        final_proposals, final_conf_scores, final_class = [], [], []

        # Predicting the final proposal coordinates `final_proposals`,         
        # confidence scores `final_conf_scores`, and the class index `final_class`.  

        # The overall steps are similar to the forward pass but now we do not need  
        # to decide the activated nor negative bounding boxes.                         
        # We threshold the conf_scores based on the threshold value `thresh`.  
        # Then, apply NMS (torchvision.ops.nms) to the filtered proposals given the  
        # threshold `nms_thresh`.                                                    
        # The class index is determined by the class with the maximal probability.   
        # Note that `final_propsals`, `final_conf_scores`, and `final_class` are all 
        # lists of B 2-D tensors.

        with torch.no_grad():
            # Feature extraction
            features = self.feat_extractor(images)

            # Grid  Generator
            grid_list = GenerateGrid(images.shape[0])

            # Prediction Network
            offsets, conf_scores, class_scores = self.pred_network(features)
            B, A, _, w_amap, h_amap = offsets.shape  # B, A, 4, H, W
            C = self.num_classes
            conf_scores = conf_scores.view(B, -1)  # B, A*H*W
            offsets = offsets.permute(0, 1, 3, 4, 2)  # B, A, H, W, 4
            class_scores = class_scores.permute(0, 2, 3, 1).reshape(B, -1, C)  # B, H*W, C

            _, most_conf_class_idx = class_scores.max(dim=-1)

            # Proposal generator
            proposals = GenerateProposal(grid_list, offsets).reshape(B, -1, 4)  # Bx(AxH'xW')x4
            # proposals = GenerateProposal(grid_list, offsets).reshape(B, -1, 4) # Bx(AxH'xW')x4
            # Thresholding and NMS
            for i in range(B):
                score_mask = torch.nonzero((conf_scores[i] > thresh)).squeeze(1)  # (AxH'xW')
                prop_before_nms = proposals[i, score_mask]
                scores_before_nms = conf_scores[i, score_mask]
                class_idx_before_nms = most_conf_class_idx[i, score_mask % (h_amap * w_amap)]
                # class_prob_before_nms = most_conf_class_prob[i, score_mask/A]

            prop_keep = torchvision.ops.nms(prop_before_nms, scores_before_nms, nms_thresh).to(images.device)
            final_proposals.append(prop_before_nms[prop_keep])
            final_conf_scores.append(scores_before_nms[prop_keep].unsqueeze(-1))
            final_class.append(class_idx_before_nms[prop_keep].unsqueeze(-1))

        return final_proposals, final_conf_scores, final_class

    def _extract_bbox_data(self, bbox_data, bbox_idx):
        """
        Inputs:
        - bbox_data: Tensor of shape (B, A, D, H, W) giving a vector of length
          D for each of A bboxes at each point in an H x W grid.
        - bbox_idx: int64 Tensor of shape (M,) giving bbox indices to extract

        Returns:
        - extracted_bboxes: Tensor of shape (M, D) giving bbox data for each
          of the bboxes specified by bbox_idx.
        """
        B, A, D, H, W = bbox_data.shape
        bbox_data = bbox_data.permute(0, 1, 3, 4, 2).contiguous().view(-1, D)
        extracted_bboxes = bbox_data[bbox_idx]
        return extracted_bboxes

    def _extract_class_scores(self, all_scores, bbox_idx):
        """
        Inputs:
        - all_scores: Tensor of shape (B, C, H, W) giving classification scores for
          C classes at each point in an H x W grid.
        - bbox_idx: int64 Tensor of shape (M,) giving the indices of bboxes at
          which to extract classification scores

        Returns:
        - extracted_scores: Tensor of shape (M, C) giving the classification scores
          for each of the bboxes specified by bbox_idx.
        """
        B, C, H, W = all_scores.shape
        A = self.num_bboxes
        all_scores = all_scores.contiguous().permute(0, 2, 3, 1).contiguous()
        all_scores = all_scores.view(B, 1, H, W, C).expand(B, A, H, W, C)
        all_scores = all_scores.reshape(B * A * H * W, C)
        extracted_scores = all_scores[bbox_idx]
        return extracted_scores
