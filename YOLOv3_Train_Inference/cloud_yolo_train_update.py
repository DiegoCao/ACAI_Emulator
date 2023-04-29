# Created by Churong Ji at 4/27/23
from model import *
from utils import *
from network import *
import torch
import socket
from torch import optim
import time

lr = 5e-2
retrain_num_epochs = 1
retrain_batch_size = 10
retrain_counter = 0
# options: ['cpu', 'gpu']
device = 'cpu'
updated_model_path = 'yolo_updated_detector.pt'

# define the server socket locally
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), 1234))
s.listen(5)
clientsocket = None

sock_established = False


def serverReceiveImg():
    # should be a BLOCKING function if not enough image received
    # returns batch img and annotation, assume img already normalized
    global sock_established
    global clientsocket
    while True and not sock_established:
        # print("Try to accept connection!")
        sock_established = True
        clientsocket, address = s.accept()
        break
    samples = receive_imgs(clientsocket)

    image_batch, box_batch, w_batch, h_batch, img_id_list = samples
    return image_batch, box_batch, w_batch, h_batch, img_id_list


def serverSendWeight(model_path):
    # send updated model parameters to the edge
    msg = modelToMessage(model_path)
    send_weights(clientsocket, msg)
    return True


def DetectionRetrain(detector, data_batch, learning_rate=3e-3,
                     lr_decay=1, num_epochs=20, device_type='cpu', **kwargs):
    if device_type == 'gpu':
        detector.to(**to_float_cuda)

    # optimizer setup
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, detector.parameters()),
        learning_rate)  # leave betas and eps by default
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                               lambda epoch: lr_decay ** epoch)

    loss_history = []
    detector.train()
    for i in range(num_epochs):
        start_t = time.time()

        images, boxes, w_batch, h_batch, _ = data_batch
        resized_boxes = coord_trans(boxes, w_batch, h_batch, mode='p2a')
        if device == 'gpu':
            images = images.to(**to_float_cuda)
            resized_boxes = resized_boxes.to(**to_float_cuda)

        loss = detector(images, resized_boxes)
        optimizer.zero_grad()
        loss.backward()
        loss_history.append(loss.item())
        optimizer.step()

        end_t = time.time()
        print('(Epoch {} / {}) loss: {:.4f} time per epoch: {:.1f}s'.format(
            i + 1, num_epochs, loss.item(), end_t - start_t))

        lr_scheduler.step()

    return loss_history


# load pretrained model
yoloDetector = SingleStageDetector()
yoloDetector.load_state_dict(torch.load('yolo_detector.pt', map_location=torch.device('cpu')))
print("Model Loaded")

# start listen to the edge, retrain and send back updated model as necessary
while True:
    print("--------------------------------------------")
    retrain_data_batch = serverReceiveImg()
    print("INFO: Incorrect image batch received from the edge")
    loss_his = DetectionRetrain(yoloDetector, retrain_data_batch, learning_rate=lr,
                                num_epochs=retrain_num_epochs, device_type=device)
    print("INFO: Retrain Round " + str(retrain_counter) + " finished")
    retrain_counter += 1
    torch.save(yoloDetector.state_dict(), updated_model_path)
    print("INFO: Model saved in ", updated_model_path)
    serverSendWeight(updated_model_path)
    print("INFO: Model params sent to edge")
