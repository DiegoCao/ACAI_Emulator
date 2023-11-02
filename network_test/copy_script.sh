kubectl cp ./client.py cpu-client-pod:/tmp/client.py -n bwtest
kubectl cp ./server.py cpu-pod-server:/tmp/server.py -n bwtest