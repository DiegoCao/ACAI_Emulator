# Created by Yuqing Qiu
# Edited by Churong Ji

# Notice: since this application reuses to same dataset for inference
#         to keep the communication between edge and cloud. You may
#         observe accuracy drop after many rounds of model retrain as
#         it overfits to certain category like cat here.

import logging
import os
import random
import shutil
import socket
import sys
import time
from model import *
from dataset import *
from utils import *
from network import *
from threading import Thread


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
    subset_buffer = send_buffer[:10]
    # update boxes
    obj_num = [len(sample[1]) for sample in subset_buffer]
    max_obj_num = max(obj_num)
    for i in range(len(subset_buffer)):
        box = subset_buffer[i][1]
        if len(box) < max_obj_num:
            diff = max_obj_num - len(box)
            width = box.size(dim=1)
            padding = torch.tensor(np.ones((diff, width)) * -1)
            send_buffer[i][1] = torch.cat((box, padding), 0)

    # stack to one tensor
    image_batch_lis = [item[0] for item in subset_buffer]
    box_batch_lis = [item[1] for item in subset_buffer]
    w_batch_lis = [item[2].item() for item in subset_buffer]
    h_batch_lis = [item[3].item() for item in subset_buffer]
    img_id_list = [item[4] for item in subset_buffer]

    image_batch = torch.stack(image_batch_lis, 0)
    box_batch = torch.stack(box_batch_lis, 0)
    w_batch = torch.tensor(w_batch_lis)
    h_batch = torch.tensor(h_batch_lis)

    # aggregate to one tuple
    send_samples = (image_batch, box_batch, w_batch, h_batch, img_id_list)
    send_buffer = []
    return send_samples


def update_model():
    update_start_time = time.perf_counter()

    global new_detector, wait_model_update, receive_model_update
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.settimeout(60.0)

    if host == "local":
        s.connect((socket.gethostname(), int(port)))
    else:
        s.connect((host, int(port)))
    send_samples = prepare_send_samples()
    bytes_send = tensorToMessage(send_samples)
    # TODO: add communication latency measurement here
    #  (but this does not mean data arrives at the cloud side?)
    #  might need send start time from edge & recv end time from cloud
    s.sendall(bytes_send)  # send all bytes
    logger_update.info("INFO: Images and annotations sent to cloud")
    receive_weights(s, model_updated_path)
    new_detector = SingleStageDetector()
    new_detector.load_state_dict(torch.load(model_updated_path, map_location=torch.device('cpu')))
    logger_update.info("INFO: Update model at edge!")
    new_detector.eval()
    logger_update.info(f"INFO: Updated model accuracy is [{get_accuracy(new_detector)[0]: .6f}]")
    wait_model_update, receive_model_update = False, True
    update_elapsed_time = time.perf_counter() - update_start_time
    logger_update.info("********************************************")
    logger_update.info(f"METRIC: Model update with cloud takes: {update_elapsed_time:.6f} seconds")


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

    logger_update.info(f"INFO: Init model accuracy is [{get_accuracy(detector)[0]: .6f}]")

    init_time = time.perf_counter() - init_start_time
    logger_inf.info(f"INFO: Init finished, taking {init_time:.6f} seconds")
    logger_update.info(f"INFO: Init finished, taking {init_time:.6f} seconds")

    # retrieve inference time every {perf_img_thresh} images to evaluate inference latency/speed
    inf_latency_img_thresh = 10
    cur_img_cnt = 0
    total_inf_time = 0

    # start inference
    global wait_model_update, receive_model_update
    wait_model_update, receive_model_update = False, False
    while True:
        for _, data_batch in enumerate(inference_loader):
            if receive_model_update:
                detector = new_detector
                receive_model_update = False

            inf_start_time = time.perf_counter()

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
                            idx_to_class[b[4].item()], b[5], b[0], b[1], b[2], b[3]))

                # add incorrect image to send_buffer when edge is not waiting for cloud update
                if not wait_model_update and\
                        not check_predictions(boxes, valid_box, resized_proposals, idx):
                    send_buffer.append([images[idx], boxes[idx], w_batch[idx], h_batch[idx], img_ids[idx]])

                cur_img_cnt += 1
                inf_end_time = time.perf_counter()
                total_inf_time += (inf_end_time - inf_start_time)
                if cur_img_cnt == inf_latency_img_thresh:
                    logger_inf.info("********************************************")
                    logger_inf.info(f"METRIC: Edge image inference in avg of {inf_latency_img_thresh} takes: "
                                 f"{total_inf_time / inf_latency_img_thresh:.6f} seconds")
                    cur_img_cnt = 0
                    total_inf_time = 0

                if len(send_buffer) == incorrect_thresh:
                    # communicate with cloud to get new model checkpoint
                    logger_update.info("--------------------------------------------")
                    logger_update.info("INFO: Reach send threshold!")
                    wait_model_update = True
                    thread = Thread(target=update_model)
                    thread.start()

            # sleep between inference requests
            # time.sleep(random.randint(1, 10) / 10)


def setup_logger(name, log_file, level=logging.INFO):
    """To set up as many loggers as you want"""

    formatter = logging.Formatter('%(asctime)s %(message)s')
    handler_1 = logging.StreamHandler(sys.stdout)
    handler_2 = logging.FileHandler(log_file)
    handler_1.setFormatter(formatter)
    handler_2.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler_1)
    logger.addHandler(handler_2)

    return logger


if __name__ == "__main__":
    args = sys.argv
    if len(args) != 5:
        print("ERROR: Incorrect Number of arguments!")
        exit(1)
    host, port, log_inf_path, log_update_path = args[1], args[2], args[3], args[4]

    if os.path.exists(log_inf_path):
        print("INFO: Old edge inf log file is deleted")
        os.remove(log_inf_path)
    if os.path.exists(log_update_path):
        print("INFO: Old edge update log file is deleted")
        os.remove(log_update_path)

    print("INFO: Edge Logs are written to ", log_inf_path, " and ", log_update_path)

    logger_inf = setup_logger('logger_inf', log_inf_path)
    logger_inf.info("INFO: The client host and port are " + host + " " + port)

    logger_update = setup_logger('logger_update', log_update_path)
    logger_update.info("INFO: The client host and port are " + host + " " + port)

    init_start_time = time.perf_counter()

    thresh = 0.8
    nms_thresh = 0.3
    cat_ratio = 1
    incorrect_thresh = 10

    if not os.path.exists("mAP"):
        os.mkdir("mAP")
    if not os.path.exists("mAP/input"):
        os.mkdir("mAP/input")
    det_dir = 'mAP/input/detection-results'
    gt_dir = 'mAP/input/ground-truth'
    model_pretrained_path = 'models/yolo_pretrained_detector_0.01cat_2500.pt'
    model_updated_path = 'models/yolo_updated_edge_detector.pt'

    val_dataset = get_pascal_voc2007_data('content', 'val')
    inference_dataset = filter_dataset_with_class(val_dataset, 'cat', cat_ratio)

    send_buffer = []
    inference()
