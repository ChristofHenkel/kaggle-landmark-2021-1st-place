
docker build  -t al_triton:0.1 - < Dockerfile

docker run --gpus all --shm-size=8g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 -v "$PWD":/workspace --ulimit stack=67108864 -ti al_triton:0.1
