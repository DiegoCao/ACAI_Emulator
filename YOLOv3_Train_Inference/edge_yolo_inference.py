# Created by Yuqing Qiu
# Edited by Churong Ji

# Notice: since this application reuses to same dataset for inference
#         to keep the communication between edge and cloud. You may
#         observe accuracy drop after many rounds of model retrain as
#         it overfits to certain category like cat here.

import os
import random
import shutil
import socket
import sys
import time
from collections import deque
from threading import Thread, Condition

import torch.cuda

from model import *
from dataset import *
from logger import setup_logger
from network import *
from utils import *


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
    update_start_time = time.perf_counter()
    # Timestamp 1
    update_timestamps = [str(update_start_time)]
    global new_detector, wait_model_update, receive_model_update
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.settimeout(60.0)
    logger_update.info("TRY TO CONNECT")
    if host == "local":
        s.connect((socket.gethostname(), int(port)))
    else:
        s.connect((host, int(port)))
    logger_update.info("CONNECT SUCCESS")
    # Timestamp 2
    update_timestamps.append(str(time.perf_counter()))
    send_samples = prepare_send_samples()
    bytes_send = tensorToMessage(send_samples)
    logger_update.info("INFO: Start sending images and annotations")
    # Timestamp 3
    update_timestamps.append(str(time.perf_counter()))
    s.sendall(bytes_send)  # send all bytes
    logger_update.info("INFO: Images and annotations sent to cloud")
    # Timestamp 4
    update_timestamps.append(str(time.perf_counter()))
    receive_weights(s, model_updated_path)
    # Timestamp 5
    update_timestamps.append(str(time.perf_counter()))
    new_detector = SingleStageDetector(edge_device)
    new_detector.load_state_dict(torch.load(model_updated_path, map_location=torch.device(edge_device)))
    logger_update.info("INFO: Update model at edge!")
    # Timestamp 6
    update_timestamps.append(str(time.perf_counter()))
    update_elapsed_time = time.perf_counter() - update_start_time
    logger_update.info(f"METRIC: Model update with cloud takes: {update_elapsed_time:.6f} seconds")
    # new_detector.eval()
    # logger_update.info(f"INFO: Updated model accuracy is [{get_accuracy(new_detector)[0]: .6f}]")
    # evaluate new model accuracy timestamp
    # update_timestamps.append(str(time.perf_counter()))
    csv_update.info(','.join(update_timestamps))
    wait_model_update, receive_model_update = False, True


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
            # final_all = torch.cat((final_proposals[idx],
            #                        final_class[idx].float(), final_conf_scores[idx]), dim=-1).cpu()
            final_all = torch.cat((final_proposals[idx],
                                   final_class[idx].float(), final_conf_scores[idx]), dim=-1)
            if edge_device == 'cpu':
                final_all = final_all.cpu()
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


def emulate_user_request(image_per_sec):
    # user inference request queue which contains the request timestamps
    global new_req_cond, req_queue
    req_interval = 1 / int(image_per_sec)
    while True:
        new_req_cond.acquire()
        req_queue.append(time.perf_counter())
        new_req_cond.notify()
        new_req_cond.release()
        time.sleep(req_interval)


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
    detector = SingleStageDetector(edge_device)
    detector.load_state_dict(torch.load(model_pretrained_path, map_location=torch.device(edge_device)))
    detector.eval()

    logger_update.info(f"INFO: Init model accuracy is [{get_accuracy(detector)[0]: .6f}]")

    init_time = time.perf_counter() - init_start_time
    logger_inf.info(f"INFO: Init finished, taking {init_time:.6f} seconds")
    logger_update.info(f"INFO: Init finished, taking {init_time:.6f} seconds")

    # retrieve inference time every {perf_img_thresh} images to evaluate inference latency/speed
    inf_latency_img_thresh = 10
    cur_img_cnt = 0
    total_inf_time = 0

    # start sending inference requests to edge
    global new_req_cond
    new_req_cond = Condition()
    global req_queue
    req_queue = deque()
    req_wait_time = 0
    queue_length = 0
    thread = Thread(target=emulate_user_request, args=(req_per_sec,))
    thread.start()
    logger_inf.info("INFO: User emulator started")

    # start inference
    global wait_model_update, receive_model_update
    wait_model_update, receive_model_update = False, False
    while True:
        for _, data_batch in enumerate(inference_loader):
            if receive_model_update:
                detector = new_detector
                receive_model_update = False

            # make sure there is requests in the queue
            new_req_cond.acquire()
            while not req_queue:
                new_req_cond.wait()
            cur_img_req_time = req_queue.popleft()
            queue_length += len(req_queue)
            new_req_cond.release()

            inf_start_time = time.perf_counter()

            images, boxes, w_batch, h_batch, img_ids = data_batch
            final_proposals, final_conf_scores, final_class = detector.inference(images, thresh=thresh,
                                                                                 nms_thresh=nms_thresh)

            # clamp on the proposal coordinates
            for idx in range(batch_size):
                torch.clamp_(final_proposals[idx][:, 0::2], min=0, max=w_batch[idx])
                torch.clamp_(final_proposals[idx][:, 1::2], min=0, max=h_batch[idx])
                valid_box = sum([1 if j != -1 else 0 for j in boxes[idx][:, 0]])
                # final_all = torch.cat((final_proposals[idx],
                #                        final_class[idx].float(), final_conf_scores[idx]), dim=-1).cpu()
                final_all = torch.cat((final_proposals[idx],
                                       final_class[idx].float(), final_conf_scores[idx]), dim=-1)
                if edge_device == 'cpu':
                    final_all = final_all.cpu()
                resized_proposals = coord_trans(final_all, w_batch[idx], h_batch[idx])

                # write results to file for evaluation (use mAP API https://github.com/Cartucho/mAP for now...)
                file_name = img_ids[idx].replace('.jpg', '.txt')
                with open(os.path.join(det_dir, file_name), 'w') as f_det, \
                        open(os.path.join(gt_dir, file_name), 'w') as f_gt:
                    for b in boxes[idx][:valid_box]:
                        f_gt.write('{} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(
                            idx_to_class[b[4].item()], b[0], b[1], b[2], b[3]))
                    for b in resized_proposals:
                        f_det.write('{} {:.6f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(
                            idx_to_class[b[4].item()], b[5], b[0], b[1], b[2], b[3]))

                # add incorrect image to send_buffer when edge is not waiting for cloud update
                if not wait_model_update and \
                        not check_predictions(boxes, valid_box, resized_proposals, idx):
                    send_buffer.append([images[idx], boxes[idx], w_batch[idx], h_batch[idx], img_ids[idx]])

                cur_img_cnt += 1
                inf_end_time = time.perf_counter()
                total_inf_time += (inf_end_time - inf_start_time)
                req_wait_time += (time.perf_counter() - cur_img_req_time)
                if cur_img_cnt == inf_latency_img_thresh:
                    logger_inf.info("--------------------------------------------")
                    avg_inf_time = total_inf_time / inf_latency_img_thresh
                    avg_wait_time = req_wait_time / inf_latency_img_thresh
                    avg_queue_len = queue_length / inf_latency_img_thresh
                    csv_inf.info(','.join([str(inf_end_time), str(avg_inf_time), str(avg_wait_time), str(avg_queue_len)]))
                    logger_inf.info(f"INFO: Edge inferences {inf_latency_img_thresh} images")
                    logger_inf.info(f"METRIC: Avg inference time for last {inf_latency_img_thresh} images takes: "
                                    f"{avg_inf_time:.6f} seconds")
                    logger_inf.info(f"METRIC: Avg wait time for last {inf_latency_img_thresh} requests is: "
                                    f"{avg_wait_time:.6f} seconds")
                    logger_inf.info(f"METRIC: Avg length of the request queue is: "
                                    f"{avg_queue_len: .2f}")
                    cur_img_cnt = 0
                    total_inf_time = 0
                    req_wait_time = 0
                    queue_length = 0

                if not wait_model_update and len(send_buffer) == incorrect_thresh:
                    # communicate with cloud to get new model checkpoint
                    logger_update.info("--------------------------------------------")
                    logger_update.info("INFO: Reach send threshold!")
                    wait_model_update = True
                    thread = Thread(target=update_model)
                    thread.start()


if __name__ == "__main__":
    args = sys.argv
    if len(args) != 9:
        print("ERROR: Incorrect Number of arguments!")
        exit(1)

    for path in args[3:7]:
        if os.path.exists(path):
            print(f"INFO: Old {path} is deleted")
            os.remove(path)

    host, port, log_inf_path, log_update_path, csv_inf_path, csv_update_path, req_per_sec, edge_id = \
        args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8]

    log_root = "logs"
    mAP_root = "mAP"
    model_root = "models"
    data_root = "content"
    if host == "local":
        log_root = os.path.join(log_root, str(edge_id))
        mAP_root = os.path.join(mAP_root, str(edge_id))
        model_root = os.path.join(model_root, str(edge_id))
        data_root = os.path.join(data_root, str(edge_id))
    if not os.path.exists(log_root):
        os.mkdir(log_root)
    if not os.path.exists(mAP_root):
        os.mkdir(mAP_root)
    if not os.path.exists(data_root):
        print("ERROR: Image folder does not exist!")
        exit(1)

    log_inf_path = os.path.join(log_root, log_inf_path)
    log_update_path = os.path.join(log_root, log_update_path)
    csv_inf_path = os.path.join(log_root, csv_inf_path)
    csv_update_path = os.path.join(log_root, csv_update_path)
    delete_if_exists = [log_inf_path, log_update_path, csv_inf_path, csv_update_path]
    for file in delete_if_exists:
        if os.path.exists(file):
            print(f"INFO: Old {file} is deleted")
            os.remove(file)
    print("INFO: Edge logs are written to ", log_inf_path, " and ", log_update_path)
    print("INFO: Edge data are written to ", csv_inf_path, " and ", csv_update_path)

    # set up loggers to log or csv file
    logger_inf = setup_logger('logger_inf', log_inf_path, "log")
    logger_inf.info("INFO: The client host and port are " + host + " " + port)
    logger_update = setup_logger('logger_update', log_update_path, "log")
    logger_update.info("INFO: The client host and port are " + host + " " + port)
    csv_inf = setup_logger('csv_inf', csv_inf_path, "csv")
    csv_update = setup_logger('csv_update', csv_update_path, "csv")

    init_start_time = time.perf_counter()

    thresh = 0.8
    nms_thresh = 0.3
    cat_ratio = 1
    incorrect_thresh = 50

    logger_inf.info("INFO: Current workload is " + req_per_sec + " requests per second")
    logger_update.info("INFO: Incorrect image send threshold is: " + str(incorrect_thresh))

    input_path = os.path.join(mAP_root, "input")
    if not os.path.exists(input_path):
        os.mkdir(input_path)
    det_dir = os.path.join(mAP_root, "input/detection-results")
    gt_dir = os.path.join(mAP_root, "input/ground-truth")

    # only supports cpu on edge side
    edge_device = 'cpu'
    logger_inf.info("INFO: Client using CPU to inference")
    # if torch.cuda.is_available():
    #     edge_device = 'cuda'
    #     logger_inf.info("INFO: Client using GPU to inference")
    # else:
    #     logger_inf.info("INFO: Client using CPU to inference")

    model_pretrained_path = os.path.join(model_root, "yolo_pretrained_detector_0.01cat_2500.pt")
    model_updated_path = os.path.join(model_root, "yolo_updated_edge_detector.pt")
    val_dataset = get_pascal_voc2007_data(data_root, 'val')
    inference_dataset = filter_dataset_with_class(val_dataset, 'cat', cat_ratio)

    send_buffer = []
    inference()
