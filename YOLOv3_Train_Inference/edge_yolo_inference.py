# Created by Yuqing Qiu
# Edited by Churong Ji

# Notice: since this application reuses to same dataset for inference
#         to keep the communication between edge and cloud. You may
#         observe accuracy drop after many rounds of model retrain as
#         it overfits to certain category like cat here.

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
import sys

args = sys.argv
host = args[1]
port = args[2]
print("the client host and port are ", host, " ", port)

thresh = 0.8
nms_thresh = 0.3
cat_ratio = 1
incorrect_thresh = 10

if not os.path.exists("mAP"):
    os.mkdir("mAP")
if not os.path.exists("mAP/input"):
    os.mkdir("mAP/input")
output_dir = 'mAP/input'
det_dir = 'mAP/input/detection-results'
gt_dir = 'mAP/input/ground-truth'
model_pretrained_path = 'models/yolo_pretrained_detector_0.01cat_2500.pt'
model_updated_path = 'models/yolo_updated_edge_detector.pt'

val_dataset = get_pascal_voc2007_data('content', 'val')
inference_dataset = filter_dataset_with_class(val_dataset, 'cat', cat_ratio)

send_buffer = []

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
if host == "local":
    s.connect((socket.gethostname(), 1234))
else:
    s.connect((host, 5001))
s.settimeout(30.0)


def get_data_loader(batch_size):
    inference_loader = pascal_voc2007_loader(inference_dataset, batch_size)
    return inference_loader


def check_predictions(boxes, valid_box, resized_proposals, idx):
    detected_classes = [idx_to_class[b[4].item()] for b in boxes[idx][:valid_box]]
    detected_classes.sort()
    gt_classes = [idx_to_class[b[4].item()] for b in resized_proposals]
    gt_classes.sort()
    if detected_classes != gt_classes:
        return False
    return True


def prepare_send_samples():
    global send_buffer
    # update boxes
    obj_num = [len(sample[1]) for sample in send_buffer]
    max_obj_num = max(obj_num)
    for i in range(len(send_buffer)):
        box = send_buffer[i][1]
        if len(box) < max_obj_num:
            diff = max_obj_num - len(box)
            width = box.size(dim=1)
            padding = torch.tensor(np.ones((diff, width)) * -1)
            send_buffer[i][1] = torch.cat((box, padding), 0)

    # stack to one tensor
    image_batch_lis = [item[0] for item in send_buffer]
    box_batch_lis = [item[1] for item in send_buffer]
    w_batch_lis = [item[2].item() for item in send_buffer]
    h_batch_lis = [item[3].item() for item in send_buffer]
    img_id_list = [item[4] for item in send_buffer]

    image_batch = torch.stack(image_batch_lis, 0)
    box_batch = torch.stack(box_batch_lis, 0)
    w_batch = torch.tensor(w_batch_lis)
    h_batch = torch.tensor(h_batch_lis)

    # aggregate to one tuple
    send_samples = (image_batch, box_batch, w_batch, h_batch, img_id_list)
    send_buffer = []
    return send_samples


def update_model():
    send_samples = prepare_send_samples()
    bytes_send = tensorToMessage(send_samples)
    s.sendall(bytes_send)  # send all bytes
    print("INFO: Images and annotations sent to cloud")
    receive_weights(s, model_updated_path)
    new_detector = SingleStageDetector()
    new_detector.load_state_dict(torch.load(model_updated_path, map_location=torch.device('cpu')))
    print("INFO: Update model at edge!")
    new_detector.eval()
    print("INFO: Updated model accuracy is ", get_accuracy(new_detector))
    return new_detector


def get_accuracy(detector):
    # get data loader
    batch_size = 1
    inference_loader = get_data_loader(batch_size)

    total = 0
    correct = 0
    correct_cat_box = 0

    # start inference
    for _, data_batch in enumerate(inference_loader):
        images, boxes, w_batch, h_batch, img_ids = data_batch
        final_proposals, final_conf_scores, final_class = detector.inference(images, thresh=thresh,
                                                                             nms_thresh=nms_thresh)
        # clamp on the proposal coordinates
        for idx in range(batch_size):
            torch.clamp_(final_proposals[idx][:, 0::2], min=0, max=w_batch[idx])
            torch.clamp_(final_proposals[idx][:, 1::2], min=0, max=h_batch[idx])
            valid_box = sum([1 if j != -1 else 0 for j in boxes[idx][:, 0]])
            final_all = torch.cat((final_proposals[idx],
                                   final_class[idx].float(), final_conf_scores[idx]), dim=-1).cpu()
            resized_proposals = coord_trans(final_all, w_batch[idx], h_batch[idx])

            # check if the predictions are incorrect
            detected_classes = [idx_to_class[b[4].item()] for b in boxes[idx][:valid_box]]
            gt_classes = [idx_to_class[b[4].item()] for b in resized_proposals]
            for i in range(min(len(gt_classes), len(detected_classes))):
                if gt_classes[i] == "cat" and detected_classes[i] == "cat":
                    correct_cat_box += 1
            detected_classes.sort()
            gt_classes.sort()
            if detected_classes == gt_classes:
                correct += 1
            total += 1

    # calculate accuracy
    accuracy = (correct / total) * 100
    return accuracy, correct_cat_box


def inference():
    if os.path.exists(det_dir):
        shutil.rmtree(det_dir)
    os.mkdir(det_dir)
    if os.path.exists(gt_dir):
        shutil.rmtree(gt_dir)
    os.mkdir(gt_dir)

    # get data loader
    batch_size = 1
    inference_loader = get_data_loader(batch_size)

    # load pre-trained model
    detector = SingleStageDetector()
    detector.load_state_dict(torch.load(model_pretrained_path, map_location=torch.device('cpu')))
    detector.eval()

    print("INFO: Pretrained accuracy is ", get_accuracy(detector))

    # start inference
    while True:
        for _, data_batch in enumerate(inference_loader):
            images, boxes, w_batch, h_batch, img_ids = data_batch
            final_proposals, final_conf_scores, final_class = detector.inference(images, thresh=thresh,
                                                                                 nms_thresh=nms_thresh)

            # clamp on the proposal coordinates
            for idx in range(batch_size):
                torch.clamp_(final_proposals[idx][:, 0::2], min=0, max=w_batch[idx])
                torch.clamp_(final_proposals[idx][:, 1::2], min=0, max=h_batch[idx])
                valid_box = sum([1 if j != -1 else 0 for j in boxes[idx][:, 0]])
                final_all = torch.cat((final_proposals[idx],
                                       final_class[idx].float(), final_conf_scores[idx]), dim=-1).cpu()
                resized_proposals = coord_trans(final_all, w_batch[idx], h_batch[idx])

                # write results to file for evaluation (use mAP API https://github.com/Cartucho/mAP for now...)
                file_name = img_ids[idx].replace('.jpg', '.txt')
                with open(os.path.join(det_dir, file_name), 'w') as f_det, \
                        open(os.path.join(gt_dir, file_name), 'w') as f_gt:
                    # print('{}: {} GT bboxes and {} proposals'.format(
                    #     img_ids[idx], valid_box, resized_proposals.shape[0]))
                    for b in boxes[idx][:valid_box]:
                        f_gt.write('{} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(
                            idx_to_class[b[4].item()], b[0], b[1], b[2], b[3]))
                    for b in resized_proposals:
                        f_det.write('{} {:.6f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(
                            idx_to_class[b[4].item()], b[5], b[0], b[1],b[2], b[3]))

                # check if the predictions are incorrect
                if not check_predictions(boxes, valid_box, resized_proposals, idx):
                    send_buffer.append([images[idx], boxes[idx], w_batch[idx], h_batch[idx], img_ids[idx]])

                # communicate with cloud to get new model checkpoint
                if len(send_buffer) == incorrect_thresh:
                    print("--------------------------------------------")
                    print("INFO: Reach send threshold!")
                    detector = update_model()

            # sleep between inference requests
            # time.sleep(random.randint(1, 10) / 10)


if __name__ == "__main__":
    inference()
