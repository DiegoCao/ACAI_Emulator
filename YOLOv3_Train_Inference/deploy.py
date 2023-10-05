import time
import yaml
import os
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

def main():
    config.load_kube_config()

    server_yaml_path = 'serverconfig.yaml'
    client_yaml_path = 'clientconfig.yaml'
    namespace = 'yolo'
    client_args_template = 'python3 /app/edge_yolo_inference.py {} 9876 logs/edge_inf.log logs/edge_update.log logs/edge_inf.csv logs/edge_update.csv'
    server_info = {}
    client_info = {}
    app_client = client.AppsV1Api()
    core_client = client.CoreV1Api()

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

    with open(client_yaml_path, 'r') as f:
        deps = yaml.safe_load_all(f)
        try:
            for dep in deps:
                name = dep['metadata']['name']
                if dep['kind'] == 'Deployment':
                    client_info['name'] = name
                    client_info['label'] = dep['metadata']['labels']['app']
                    client_info['existing_pods'] = []
                    dep['spec']['template']['spec']['containers'][0]['args'][1] = client_args_template.format(server_info['IP'])
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
                print("Clinet pod name:", pod.metadata.name)
                print("Client status:", pod.status.phase, pod.status.pod_ip)
                client_info['pod_name'] = pod.metadata.name
                client_info['IP'] = pod.status.pod_ip
                launched = True
                break
        time.sleep(5)

if __name__ == '__main__':
    main()