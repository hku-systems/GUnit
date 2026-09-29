#!/bin/bash

docker run -it --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus all --name rapid --cap-add=SYS_ADMIN --runtime=nvidia -u $(id -u):$(id -g)  -v ~/:/home/rapid rapid bash
