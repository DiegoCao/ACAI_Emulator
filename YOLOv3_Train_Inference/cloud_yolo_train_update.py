# Created by Churong Ji at 4/27/23

import logging
import os
import socket
import sys
import time
from torch import optim
from model import *
from utils import *
from network import *
from threading import Thread


def serverReceiveImg():
    # should be a BLOCKING function if not enough image received
    # returns batch img and annotation, assume img already normalized

    # TODO: Ask - usage of the below commented lines?
    global clientsocket
    # while True and not sock_established:
    #     # print("Try to accept connection!")
    #     sock_established = True
    clientsocket, address = s.accept()
    samples = receive_imgs(clientsocket)
    image_batch, box_batch, w_batch, h_batch, img_id_list = samples
    return image_batch, box_batch, w_batch, h_batch, img_id_list


def serverSendWeight(model_path):
    # send updated model parameters to the edge
    msg = modelToMessage(model_path)
    send_weights(clientsocket, msg)
    return True


def DetectionRetrain(detector, learning_rate=3e-3,
                     learning_rate_decay=1, num_epochs=20, device_type='cpu'):
    if device_type == 'gpu':
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
        logging.info("--------------------------------------------")
        retrain_data_batch = serverReceiveImg()
        logging.info("INFO: Incorrect image batch received from the edge")

        retrain_start_time = time.perf_counter()

        for i in range(num_epochs):
            start_t = time.time()

            images, boxes, w_batch, h_batch, _ = retrain_data_batch
            resized_boxes = coord_trans(boxes, w_batch, h_batch, mode='p2a')
            if device == 'gpu':
                images = images.to(**to_float_cuda)
                resized_boxes = resized_boxes.to(**to_float_cuda)

            loss = detector(images, resized_boxes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            end_t = time.time()
            logging.info('(Epoch {} / {}) loss: {:.4f} time per epoch: {:.1f}s'.format(
                i + 1, num_epochs, loss.item(), end_t - start_t))

            lr_scheduler.step()

        logging.info("INFO: Retrain Round " + str(retrain_counter) + " finished")

        retrain_time = time.perf_counter() - retrain_start_time
        logging.info("********************************************")
        logging.info(f"METRIC: Cloud model refine takes: {retrain_time:.6f} seconds")

        retrain_counter += 1
        torch.save(yoloDetector.state_dict(), updated_model_path)
        logging.info("INFO: Model saved in " + updated_model_path)
        # TODO: may can add communication latency measurement here as the edge part
        serverSendWeight(updated_model_path)
        logging.info("INFO: Model params sent to edge")


if __name__ == "__main__":
    args = sys.argv
    if len(args) != 3:
        print("ERROR: Incorrect Number of arguments!")
        exit(1)
    port, log_file = args[1], args[2]

    if os.path.exists(log_file):
        print("INFO: Old cloud log file is deleted")
        os.remove(log_file)
    print("INFO: Cloud Logs are written to ", log_file)

    targets = logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)
    logging.basicConfig(format='%(message)s', level=logging.INFO, handlers=targets)

    logging.info("INFO: The server port is " + port)

    init_start_time = time.perf_counter()

    lr = 1e-3
    lr_decay = 0.8
    retrain_num_epochs = 1
    retrain_batch_size = 10
    # options: ['cpu', 'gpu']
    device = 'cpu'
    updated_model_path = 'models/yolo_updated_detector.pt'

    # define the server socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(200.0)

    s.bind((socket.gethostname(), int(port)))
    s.listen(5)
    clientsocket = None
    # clientsocket, address = s.accept()


    # load pretrained model
    yoloDetector = SingleStageDetector()
    yoloDetector.load_state_dict(torch.load('models/yolo_detector.pt', map_location=torch.device('cpu')))
    logging.info("INFO: Model Loaded")

    init_time = time.perf_counter() - init_start_time
    logging.info(f"INFO: Init finished, taking {init_time:.6f} seconds")

    # start listen to the edge, retrain and send back updated model as necessary
    DetectionRetrain(yoloDetector, learning_rate=lr, num_epochs=retrain_num_epochs, device_type=device)
