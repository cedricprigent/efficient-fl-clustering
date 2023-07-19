# G5K FL deployments

## Getting started

### Installing E2Clab on Grid'5000
[E2Clab documentation](https://e2clab.gitlabpages.inria.fr/e2clab/index.html)

From a Grid'5000 front-end (for instance lille.grid5000.fr)
```shell
# from the G5K frontend
$ ssh lille.grid5000.fr
$ cd git/
$ virtualenv -p python3 venv
$ source venv/bin/activate
$ git clone https://gitlab.inria.fr/E2Clab/e2clab.git
$ cd e2clab
$ pip install -U -e .
```

### Cloning this repository OR Copying folder from your laptop to G5K front-end
**Cloning the repository**
```shell
# from the G5K frontend
$ git clone https://gitlab.inria.fr/Kerdata/Kerdata-Codes/fl-base.git
```

**Pull docker image**
```shell
docker pull nvidia/cuda:11.2.2-base-ubuntu20.04
docker save nvidia/cuda:11.2.2-base-ubuntu20.04 > .fl-clustering/artifacts/docker-image.tar
```

**Copying folder from laptop to G5K front-end**
```shell
# or copying from your laptop
$ scp -r fl-clustering username@access.grid5000.fr:lille
```

### Adapting workflow file for your homedir
Modify the saving directory for log files 
```shell
$ nano fl-clustering/scenario/grid5000/workflow.yaml
```
```yaml
# Server
- hosts: cloud.FLServer.*
  vars:
    container_name: "fl-container"
    log_dir: "{{ container_name }}:/app/fl_logs/"
    save_dir: "/home/[your_username]/runs/"
```

### Launching experiments
Reserving nodes and installing services
```shell
# from the G5K frontend
$ cd fl-base
$ e2clab layers-services scenario/grid5000 artifacts
```

Configuring network
```shell
# from the G5K frontend
$ e2clab network scenario/grid5000
```

Workflow
```shell
# from the G5K frontend
# Prepare phase: build and run docker image, download dataset, prepare client partitions
$ e2clab workflow scenario/grid5000 prepare
# Launch phase: start server and clients
$ e2clab workflow scenario/grid5000 launch
# Finalize phase (when experiments are finished): retrieve log files from docker container and save them to your G5K homedir
$ e2clab workflow scenario/grid5000 finalize
```

### Connecting to server node and inspecting log file
```shell
# from the G5K frontend
$ ssh root@chifflet-[server_id]
$ sudo docker exec -it fl-container bash
$ cat log_traces.log
```

### Visualize metrics from your laptop with TensorBoard
```shell
# from your laptop
$ bash tunnel.sh chifflet-[server_id]
```

## Other commands and useful resources

### Killing the experiment
```shell
# from the G5K frontend
$ oarstat -u #show current jobs
$ oardel [job_id]
```

### Hardware resources (cluster information)
https://www.grid5000.fr/w/Hardware

### Cluster usage
https://intranet.grid5000.fr/oar/Rennes/drawgantt-svg/

https://intranet.grid5000.fr/oar/Grenoble/drawgantt-svg/

https://intranet.grid5000.fr/oar/Lyon/drawgantt-svg/

https://intranet.grid5000.fr/oar/Lille/drawgantt-svg/
