import pickle


def encodeBytes(filepath: str):
    with open(filepath, 'rb') as fp:
        obj = fp.read()
        return obj


def tensorToMessage(torchtensor):
    t_obj = pickle.dumps(torchtensor)
    HEADERSIZE = 10
    return bytes(f"{len(t_obj):<{HEADERSIZE}}", 'utf-8') + t_obj


def encodeMsgsize(img):
    HEADERSIZE = 10
    return bytes(f"{len(img):<{HEADERSIZE}}", 'utf-8') + img


def modelToMessage(modelpath):
    model_bytes = encodeBytes(modelpath)
    msg = encodeMsgsize(model_bytes)
    return msg


def send_weights(socket, weightbytes):
    # print("start send weights with length ", len(weightbytes))
    socket.sendall(weightbytes)
    # print("send weights done")


def receive_imgs(s):
    # print("start receiveing images")
    HEADERSIZE = 10
    while True:
        full_msg = b''
        new_msg = True
        while True:
            msg = s.recv(1024 * 1024)
            if new_msg:
                # print("new msg len:", msg[:HEADERSIZE])
                msglen = int(msg[:HEADERSIZE])
                new_msg = False
            # print(f"full message length: {msglen}")
            full_msg += msg
            # print("the length of fullmessage is ", len(full_msg)) # need to get wait
            if len(full_msg) - HEADERSIZE == msglen:
                obj = pickle.loads(full_msg[HEADERSIZE:])
                # recv_wt = open(file_write_path, "wb")
                return obj


def receive_weights(s, newpath):
    # print("start receiveing weights")
    HEADERSIZE = 10
    while True:
        full_msg = b''
        new_msg = True
        while True:
            msg = s.recv(1024 * 1024)
            if new_msg:
                # print("new msg len:", msg[:HEADERSIZE])
                msglen = int(msg[:HEADERSIZE])
                new_msg = False
            full_msg += msg
            if len(full_msg) - HEADERSIZE == msglen:
                weight_bytes = full_msg[HEADERSIZE:]
                with open(newpath, "wb+") as fp:
                    fp.write(weight_bytes)
                return weight_bytes
