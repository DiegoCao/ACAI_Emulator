# Created by Churong Ji at 4/27/23

import os
import socket
import sys
import time
from logger import setup_logger
from model import *
from network import *
from utils import *
from torch import optim


def serverReceiveImg():
    # should be a BLOCKING function if not enough image received
    # returns batch img and annotation, assume img already normalized
    global client_socket
    client_socket, address = server_socket.accept()
    samples = receive_imgs(client_socket)
    image_batch, box_batch, w_batch, h_batch, img_id_list = samples
    return image_batch, box_batch, w_batch, h_batch, img_id_list


def serverSendWeight(model_path):
    # send updated model parameters to the edge
    msg = modelToMessage(model_path)
    logger_log.info(f"METRIC: Cloud starts sending model to edge")
    # TODO: Timestamp
    cloud_timestamps.append(str(time.perf_counter()))
    send_weights(client_socket, msg)
    # TODO: Timestamp
    cloud_timestamps.append(str(time.perf_counter()))
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
        logger_log.info("--------------------------------------------")
        retrain_data_batch = serverReceiveImg()
        logger_log.info("INFO: Incorrect image batch received from the edge")
        # TODO: Timestamp
        global cloud_timestamps
        cloud_timestamps = [str(time.perf_counter())]

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
        serverSendWeight(updated_model_path)
        logger_log.info("INFO: Model params sent to edge")
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
    client_socket = None

    # load pretrained model
    yoloDetector = SingleStageDetector()
    yoloDetector.load_state_dict(torch.load('models/yolo_detector.pt', map_location=torch.device('cpu')))
    logger_log.info("INFO: Model Loaded")

    init_time = time.perf_counter() - init_start_time
    logger_log.info(f"INFO: Init finished, taking {init_time:.6f} seconds")

    # start listen to the edge, retrain and send back updated model as necessary
    DetectionRetrain(yoloDetector, learning_rate=lr, num_epochs=retrain_num_epochs, device_type=device)
