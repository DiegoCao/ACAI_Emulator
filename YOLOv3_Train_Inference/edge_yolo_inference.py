# Created by Yuqing Qiu
import torch
import time
import shutil
import os
import random
import socket
from model import *
from dataset import *
from utils import *
from network import *


def comm_with_cloud(samples):
    return 'yolo_pretrained_detector.pt'


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), 1234))

# get dataset
val_dataset = get_pascal_voc2007_data('content', 'val')
inference_dataset = filter_dataset_with_class(val_dataset, 'cat', 1)
inference_loader = pascal_voc2007_loader(inference_dataset, 1)

# load pre-trained model
detector = SingleStageDetector()
detector.load_state_dict(torch.load('yolo_pretrained_detector.pt', map_location=torch.device('cpu')))

# inference images
thresh=0.8
nms_thresh=0.3
output_dir='mAP/input'
incorrect_thresh = 10
incorrect_samples = []
detector.eval()
start_t = time.time()

if output_dir is not None:
    det_dir = 'mAP/input/detection-results'
    gt_dir = 'mAP/input/ground-truth'
    if os.path.exists(det_dir):
        shutil.rmtree(det_dir)
    os.mkdir(det_dir)
    if os.path.exists(gt_dir):
        shutil.rmtree(gt_dir)
    os.mkdir(gt_dir)


for iter_num, data_batch in enumerate(inference_loader):
    images, boxes, w_batch, h_batch, img_ids = data_batch
    final_proposals, final_conf_scores, final_class = detector.inference(images, thresh=thresh, nms_thresh=nms_thresh)

    # clamp on the proposal coordinates
    batch_size = len(images)
    for idx in range(batch_size):
        torch.clamp_(final_proposals[idx][:, 0::2], min=0, max=w_batch[idx])
        torch.clamp_(final_proposals[idx][:, 1::2], min=0, max=h_batch[idx])
        img = images[idx]
        valid_box = sum([1 if j != -1 else 0 for j in boxes[idx][:, 0]])
        final_all = torch.cat((final_proposals[idx], \
                               final_class[idx].float(), final_conf_scores[idx]), dim=-1).cpu()
        resized_proposals = coord_trans(final_all, w_batch[idx], h_batch[idx])

        # write results to file for evaluation (use mAP API https://github.com/Cartucho/mAP for now...)
        if output_dir is not None:
            file_name = img_ids[idx].replace('.jpg', '.txt')
            with open(os.path.join(det_dir, file_name), 'w') as f_det, \
                open(os.path.join(gt_dir, file_name), 'w') as f_gt:
                # print('{}: {} GT bboxes and {} proposals'.format(img_ids[idx], valid_box, resized_proposals.shape[0]))
                for b in boxes[idx][:valid_box]:
                    f_gt.write('{} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(idx_to_class[b[4].item()], b[0], b[1], b[2], b[3]))
                for b in resized_proposals:
                    f_det.write('{} {:.6f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(idx_to_class[b[4].item()], b[5], b[0], b[1], b[2], b[3]))
        else:
            data_visualizer(img, idx_to_class, boxes[idx][:valid_box], resized_proposals)

        # check if the sample is incorrect
        detected_classes = [idx_to_class[b[4].item()] for b in boxes[idx][:valid_box]]
        detected_classes.sort()
        gt_classes = [idx_to_class[b[4].item()] for b in resized_proposals]
        gt_classes.sort()
        if detected_classes != gt_classes:
            incorrect_samples.append((images[idx], boxes[idx], w_batch[idx], h_batch[idx], img_ids[idx]))
            # communicate with cloud to get new model checkpoint
            if len(incorrect_samples) == incorrect_thresh:
                # large_tensor = torch.cat(incorrect_samples)
                print("need to send message!")
                bytes_send = tensorToMessage(incorrect_samples)
                s.sendall(bytes_send)  # send all bytes 
                weight_bytes = receive_weights(s)
                # TODO: add save pt with receive byte weights
                    
                new_model_pt = comm_with_cloud(incorrect_samples)
                detector.load_state_dict(torch.load(new_model_pt, map_location=torch.device('cpu')))
                incorrect_samples = []

    # sleep between inference requests
    # time.sleep(random.randint(1, 10) / 10)

end_t = time.time()
print('Total inference time: {:.1f}s'.format(end_t-start_t))