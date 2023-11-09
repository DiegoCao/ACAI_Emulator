import time
import yaml
import sys
import os
import shutil
import json
import pathlib
import re
from kubernetes import client, config
from kubernetes.stream import stream

def deployment_exists(namespace, deployment_name):
    config.load_kube_config()
    v1 = client.AppsV1Api()
    resp = v1.list_namespaced_deployment(namespace=namespace)
    for i in resp.items:
        if i.metadata.name == deployment_name:
            return True
    return False

def service_exists(namespace, service_name):
    config.load_kube_config()
    v1 = client.CoreV1Api()
    resp = v1.list_namespaced_service(namespace=namespace)
    for i in resp.items:
        if i.metadata.name == service_name:
            return True
    return False

def generate_yamls(template_yaml_path, config_path):
    yamls = []
    config = json.load(open(config_path))
    print(config)
    return yamls

def launch_server(server_yaml_path, namespace, core_client, app_client):
    server_info = {}

    with open(server_yaml_path, 'r') as f:
        deps = yaml.safe_load_all(f)
        try:
            for dep in deps:
                name = dep['metadata']['name']
                if dep['kind'] == 'Deployment':
                    server_info['name'] = name
                    server_info['label'] = dep['metadata']['labels']['app']
                    server_info['existing_pods'] = []
                    if deployment_exists(namespace, name):
                        server_pods = core_client.list_namespaced_pod(namespace, label_selector="app=" + server_info['label']).items
                        for pod in server_pods:
                            server_info['existing_pods'].append(pod.status.pod_ip)
                        resp = app_client.delete_namespaced_deployment(name=name, namespace=namespace)
                    resp = app_client.create_namespaced_deployment(body=dep, namespace=namespace)
                    print(f"Deployment created. Status='{resp.metadata.name}'")
                elif dep['kind'] == 'Service':
                    if service_exists(namespace, name):
                        resp = core_client.delete_namespaced_service(name=name, namespace=namespace)
                    resp = core_client.create_namespaced_service(body = dep, namespace=namespace)
                    print(f"Service created. Status='{resp.metadata.name}'")
        except client.rest.ApiException as e:
            print("Exception when calling AppsV1Api->create_namespaced_replica_set: %s\n" % e)

    time.sleep(10)

    launched = False
    while not launched:
        server_pods = core_client.list_namespaced_pod(namespace, label_selector="app=" + server_info['label']).items
        for pod in server_pods:
            if pod.status.pod_ip not in server_info['existing_pods'] and pod.status.phase == 'Running':
                print("Server pod name:", pod.metadata.name)
                print("Server status:", pod.status.phase, pod.status.pod_ip)
                server_info['pod_name'] = pod.metadata.name
                server_info['IP'] = pod.status.pod_ip
                launched = True
                break
        time.sleep(5)

    return server_info

def launch_client(server_info, client_yaml_path, 
                    namespace, core_client, app_client, client_id):
    client_info = {}

    with open(client_yaml_path, 'r') as f:
        deps = yaml.safe_load_all(f)
        try:
            for dep in deps:
                dep['metadata']['name'] = dep['metadata']['name'].format(client_id)
                name = dep['metadata']['name']
                if dep['kind'] == 'Deployment':
                    client_info['name'] = name
                    client_info['label'] = dep['metadata']['labels']['app']
                    client_info['existing_pods'] = []
                    dep['spec']['template']['spec']['containers'][0]['args'][1] = \
                          dep['spec']['template']['spec']['containers'][0]['args'][1].format(server_info['IP'], client_id)
                    for volume in dep['spec']['template']['spec']['volumes']:
                        volume['nfs']['path'] = volume['nfs']['path'] + "/" + str(client_id)
                        user_path = re.sub('/users/', '/h/', volume['nfs']['path'])
                        pathlib.Path(user_path).mkdir(parents=True, exist_ok=True)
                        os.chmod(user_path, 0o777)
                        if volume['name'] == 'dataset-volume':
                            if not os.listdir(user_path):
                                shutil.copy2('/h/churongj/ACAI_Emulator/YOLOv3_Train_Inference/content/VOCtrainval_06-Nov-2007.tar', 
                                                user_path)
                    if deployment_exists(namespace, name):
                        client_pods = core_client.list_namespaced_pod(namespace, label_selector="app=" + client_info['label']).items
                        for pod in client_pods:
                            client_info['existing_pods'].append(pod.status.pod_ip)
                        resp = app_client.delete_namespaced_deployment(name=name, namespace=namespace)
                    resp = app_client.create_namespaced_deployment(body=dep, namespace=namespace)
                    print(f"Deployment created. Status='{resp.metadata.name}'")
                elif dep['kind'] == 'Service':
                    if service_exists(namespace, name):
                        resp = core_client.delete_namespaced_service(name=name, namespace=namespace)
                    resp = core_client.create_namespaced_service(body = dep, namespace=namespace)
                    print(f"Service created. Status='{resp.metadata.name}'")
        except client.rest.ApiException as e:
            print("Exception when calling AppsV1Api->create_namespaced_replica_set: %s\n" % e)

    launched = False
    while not launched:
        client_pods = core_client.list_namespaced_pod(namespace, label_selector="app=" + client_info['label']).items
        for pod in client_pods:
            if pod.status.pod_ip not in client_info['existing_pods'] and pod.status.phase == 'Running':
                print("Client pod name:", pod.metadata.name)
                print("Client status:", pod.status.phase, pod.status.pod_ip)
                client_info['pod_name'] = pod.metadata.name
                client_info['IP'] = pod.status.pod_ip
                launched = True
                break
        time.sleep(5)

    return client_info

def main(path, client_num):
    config.load_kube_config()

    server_yaml_path = 'serverconfig.yaml'
    client_yaml_path = 'clientconfig.yaml'
    namespace = 'yolo'
    app_client = client.AppsV1Api()
    core_client = client.CoreV1Api()
    configuration = client.Configuration()

    template_yaml_path = 'yaml_templates/config.yaml'
    config_path = 'user_config.json'
    yamls = generate_yamls(template_yaml_path, config_path)

    server_info = launch_server(server_yaml_path, namespace, core_client, app_client)
    for i in range(client_num):
        client_info = launch_client(server_info, client_yaml_path, namespace, core_client, app_client, i)

    time.sleep(20)

    source_folder = '/h/churongj/ACAI_Emulator/YOLOv3_Train_Inference/logs/'
    with open(path + '/consumption.csv', 'w') as f:
        f.write('server_cpu,server_memory,client_cpu,client_memory\n')
    while True:
        api = client.CustomObjectsApi()
        k8s_pods = api.list_namespaced_custom_object(group="metrics.k8s.io",version="v1beta1", namespace=namespace, plural="pods")
        data = [''] * 4
        for pod in k8s_pods['items']:
            container = pod['containers'][0]
            if container['name'] == 'yolo-demo-server':
                data[0] = container['usage']['cpu'][:-1]
                data[1] = container['usage']['memory'][:-2]
            elif container['name'] == 'yolo-demo-client':
                data[2] = container['usage']['cpu'][:-1]
                data[3] = container['usage']['memory'][:-2]
        print(','.join(data))
        with open(path + '/consumption.csv', 'a') as f:
            f.write(','.join(data) + '\n')
        
        def recursive_copy(src_folder, dest_folder):
            for file_name in os.listdir(src_folder):
                source = src_folder + file_name
                if os.path.isfile(source):
                    shutil.copy2(source, dest_folder)
                else:
                    sub_dest_folder = dest_folder + '/' + file_name
                    pathlib.Path(sub_dest_folder).mkdir(parents=True, exist_ok=True)
                    os.chmod(sub_dest_folder, 0o777)
                    recursive_copy(source + '/', sub_dest_folder)

        recursive_copy(source_folder, path)
        time.sleep(5)

if __name__ == '__main__':
    path = sys.argv[1]
    client_num = int(sys.argv[2])
    main(path, client_num)