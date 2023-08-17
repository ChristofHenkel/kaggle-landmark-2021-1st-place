docker build  -t al_triton_sdk:0.1 - < Dockerfile_sdk

docker run --shm-size=1g --net host -v "$PWD":/workspace  -ti al_triton_sdk:0.1 /bin/bash
