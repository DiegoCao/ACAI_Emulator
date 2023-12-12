# Emulator_App

The project of emulator and the sample workload: an APP of Realtime Object detection.

# Deployment

To deploy an cloud application using the Cloud-Edge Emulator, you first need to create an experiment configuration file. A sample config file can be found at [Title](YOLOv3_Train_Inference/user_config.json). The information to contain includes machines involved in the application deployment and the connections between them with the constraints specified.

To launch an experiment on our sample workload YoloV3, you can use the command
```
cd YOLOv3_Train_Inference
mkdir logs
python3 deploy.py logs user_config.json
```
A more generalized command is
```
python3 YOLOv3_Train_Inference/deploy.py <path to the directory to store logs> <path to the config file>
```

The logs will be copied from its NFS mounted location to the directory assigned by the user every 10 seconds. The realtime CPU and memory consumption of each machine specified in user-input config file will be recorded in the log folder as well.

# Sample Application

<TODO>