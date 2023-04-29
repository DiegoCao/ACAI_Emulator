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

# inference images
thresh=0.8
nms_thresh=0.3
incorrect_thresh = 10

output_dir='mAP/input'
det_dir = 'mAP/input/detection-results'
gt_dir = 'mAP/input/ground-truth'

send_buffer = []

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), 1234))
s.settimeout(10.0)

def comm_with_cloud(samples):
    return 'yolo_pretrained_detector.pt'

def get_data_loader():
    # get dataset
    val_dataset = get_pascal_voc2007_data('content', 'val')
    inference_dataset = filter_dataset_with_class(val_dataset, 'cat', 1)
    inference_loader = pascal_voc2007_loader(inference_dataset, 1)
    return inference_loader

def check_predictions(boxes, valid_box, resized_proposals, idx, images, w_batch, h_batch, img_ids):
    detected_classes = [idx_to_class[b[4].item()] for b in boxes[idx][:valid_box]]
    detected_classes.sort()
    gt_classes = [idx_to_class[b[4].item()] for b in resized_proposals]
    gt_classes.sort()
    if detected_classes != gt_classes:
        send_buffer.append([images[idx], boxes[idx], w_batch[idx], h_batch[idx], img_ids[idx]])

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

    print("img batch shape: ", image_batch.shape)
    print("box batch shape: ", box_batch.shape)
    print("w_batch shape: ", w_batch.shape)
    print("h_batch shape: ", h_batch.shape)
    print("img_id length: ", len(img_id_list))
    send_samples = (image_batch, box_batch, w_batch, h_batch, img_id_list)
    send_buffer = []
    return send_samples

def update_model():
    send_samples = prepare_send_samples()
    bytes_send = tensorToMessage(send_samples)
    s.sendall(bytes_send)  # send all bytes 
    weight_bytes = receive_weights(s)
    # TODO: add save pt with receive byte weights
        
    new_model_pt = comm_with_cloud(send_samples)
    new_detector = SingleStageDetector()
    new_detector.load_state_dict(torch.load(new_model_pt, map_location=torch.device('cpu')))
    
    return new_detector

def inference():
    if os.path.exists(det_dir):
        shutil.rmtree(det_dir)
    os.mkdir(det_dir)
    if os.path.exists(gt_dir):
        shutil.rmtree(gt_dir)
    os.mkdir(gt_dir)

    # get data loader
    inference_loader = get_data_loader()

    # load pre-trained model
    detector = SingleStageDetector()
    detector.load_state_dict(torch.load('yolo_pretrained_detector.pt', map_location=torch.device('cpu')))
    detector.eval()

    # start inference
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
            file_name = img_ids[idx].replace('.jpg', '.txt')
            with open(os.path.join(det_dir, file_name), 'w') as f_det, \
                open(os.path.join(gt_dir, file_name), 'w') as f_gt:
                # print('{}: {} GT bboxes and {} proposals'.format(img_ids[idx], valid_box, resized_proposals.shape[0]))
                for b in boxes[idx][:valid_box]:
                    f_gt.write('{} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(idx_to_class[b[4].item()], b[0], b[1], b[2], b[3]))
                for b in resized_proposals:
                    f_det.write('{} {:.6f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(idx_to_class[b[4].item()], b[5], b[0], b[1], b[2], b[3]))

        # check if the predictions are incorrect
        check_predictions(boxes, valid_box, resized_proposals, idx, images, w_batch, h_batch, img_ids)

        # communicate with cloud to get new model checkpoint
        if len(send_buffer) == incorrect_thresh:
            print("need to send message!")
            detector = update_model()

        # sleep between inference requests
        time.sleep(random.randint(1, 10) / 10)


if __name__ == "__main__":
    inference()