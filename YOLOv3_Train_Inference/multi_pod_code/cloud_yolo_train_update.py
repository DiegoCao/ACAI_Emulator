# Created by Churong Ji at 4/27/23

import os
import socket
import sys
import threading
import time
import torch

from logger import setup_logger
from model import *
from network import *
from utils import *
from torch import optim


def serverReceiveImg(client_socket):
    # assume img already normalized
    samples = receive_imgs(client_socket)
    global img_buf_cond, incorrect_img_buf
    img_buf_cond.acquire()
    incorrect_img_buf.append(samples)
    # TODO: box size can vary - decide whether to concat for gpu and more realistic training
    # if incorrect_img_buf:
    #     # image_batch, box_batch, w_batch, h_batch, img_id_list = samples
    #     # print(image_batch.size())
    #     print("SAMPLE SIZE: ", len(samples))
    #     for i in range(len(samples)):
    #         print(incorrect_img_buf[i].size(), samples[i].size())
    #         temp = torch.cat([incorrect_img_buf[i], samples[i]], dim=0)
    #         incorrect_img_buf[i] = temp
    # else:
    #     incorrect_img_buf = list(samples)

    logger_log.info("INFO: Receive incorrect img batch from one client")
    clients_lock.acquire()
    global next_clients
    next_clients.append(client_socket)
    clients_lock.release()
    img_buf_cond.notify()
    img_buf_cond.release()


def serverSendWeight(client_socket, msg):
    # send updated model parameters to the edge
    send_weights(client_socket, msg)
    client_socket.close()
    return True


def accept_clients():
    while True:
        client_socket, addr = server_socket.accept()
        logger_log.info("INFO: Accept a new client socket connection")
        client_receive_thread = threading.Thread(target=serverReceiveImg, args=(client_socket,))
        client_receive_thread.start()

def DetectionRetrain(detector, device_type, learning_rate=3e-3,
                     learning_rate_decay=1, num_epochs=20):

    if device_type == 'cuda':
        # ship model to GPU
        detector.to(**to_float_cuda)

    # optimizer setup
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, detector.parameters()),
        lr=learning_rate)  # leave betas and eps by default
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                               lambda epoch: learning_rate_decay ** epoch)

    detector.train()
    retrain_counter = 0
    while True:
        logger_log.info("--------------------------------------------")
        img_buf_cond.acquire()
        global incorrect_img_buf, cur_clients, next_clients
        while not incorrect_img_buf:
        # while len(incorrect_img_buf) < 2:
            img_buf_cond.wait()
        # TODO: IF DEEPCOPY NEEDED?
        clients_lock.acquire()
        retrain_data_batches = incorrect_img_buf
        incorrect_img_buf = []
        cur_clients = next_clients
        next_clients = []
        clients_lock.release()
        img_buf_cond.release()

        logger_log.info("INFO: Retrain using " + str(len(retrain_data_batches) *
                                                     retrain_data_batches[0][0].size(dim=0)) + " incorrect images")
        # TODO: Timestamp
        cloud_timestamps = [str(time.perf_counter())]

        retrain_start_time = time.perf_counter()
        # TODO: consider change to one batch training
        for cur_batch in retrain_data_batches:
            for i in range(num_epochs):
                start_t = time.time()

                images, boxes, w_batch, h_batch, _ = cur_batch
                resized_boxes = coord_trans(boxes, w_batch, h_batch, mode='p2a')
                if device == 'gpu':
                    images = images.to(**to_float_cuda)
                    resized_boxes = resized_boxes.to(**to_float_cuda)

                loss = detector(images, resized_boxes)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                end_t = time.time()
                logger_log.info('(Epoch {} / {}) loss: {:.4f} time per epoch: {:.1f}s'.format(
                    i + 1, num_epochs, loss.item(), end_t - start_t))

                lr_scheduler.step()

        logger_log.info("INFO: Retrain Round " + str(retrain_counter) + " finished")

        retrain_time = time.perf_counter() - retrain_start_time
        logger_log.info(f"METRIC: Cloud model refine takes: {retrain_time:.6f} seconds")
        # TODO: Timestamp
        cloud_timestamps.append(str(time.perf_counter()))

        retrain_counter += 1
        torch.save(yoloDetector.state_dict(), updated_model_path)
        logger_log.info("INFO: Model saved in " + updated_model_path)
        # TODO: Timestamp
        cloud_timestamps.append(str(time.perf_counter()))

        logger_log.info("INFO: Cloud starts sending model to edge")
        msg = modelToMessage(updated_model_path)
        clients_lock.acquire()
        for client_socket in cur_clients:
            client_send_thread = threading.Thread(target=serverSendWeight, args=(client_socket, msg,))
            client_send_thread.start()
        logger_log.info("INFO: " + str(len(cur_clients)) + " threads spawned to send model params sent to edges")
        clients_lock.release()
        logger_csv.info(','.join(cloud_timestamps))


if __name__ == "__main__":
    args = sys.argv
    if len(args) != 4:
        print("ERROR: Incorrect Number of arguments!")
        exit(1)
    port, log_file, csv_file = args[1], args[2], args[3]

    for path in args[2:]:
        if os.path.exists(path):
            print(f"INFO: Old {path} is deleted")
            os.remove(path)
    print("INFO: Cloud logs are written to ", log_file)
    print("INFO: Cloud data are written to ", csv_file)

    # set up loggers to log or csv file
    logger_log = setup_logger('logger_log', log_file, "log")
    logger_csv = setup_logger('logger_csv', csv_file, "csv")
    logger_log.info("INFO: The server port is " + port)

    init_start_time = time.perf_counter()

    lr = 1e-3
    lr_decay = 0.8
    retrain_num_epochs = 1
    retrain_batch_size = 10
    # options: ['cpu', 'gpu']
    device = 'cpu'
    updated_model_path = 'models/yolo_updated_detector.pt'

    # define the server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # server_socket.settimeout(200.0)

    server_socket.bind((socket.gethostname(), int(port)))
    server_socket.listen(5)

    # load pretrained model
    yoloDetector = SingleStageDetector()
    yoloDetector.load_state_dict(torch.load('models/yolo_detector.pt', map_location=torch.device('cpu')))
    logger_log.info("INFO: Model Loaded")

    init_time = time.perf_counter() - init_start_time
    logger_log.info(f"INFO: Init finished, taking {init_time:.6f} seconds")

    cur_clients, next_clients = [], []
    clients_lock = threading.Lock()
    incorrect_img_buf = []
    img_buf_cond = threading.Condition()
    accept_thread = threading.Thread(target=accept_clients)
    accept_thread.start()

    # start listen to the edge, retrain and send back updated model as necessary
    DetectionRetrain(yoloDetector, learning_rate=lr, num_epochs=retrain_num_epochs, device_type=device)
