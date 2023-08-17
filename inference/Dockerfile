FROM nvcr.io/nvidia/tritonserver:23.04-py3

RUN apt-get update
RUN python3 -m pip install --upgrade pip
RUN pip install torch==1.9.1
#RUN pip install pytorch==1.9
RUN pip install torchvision==0.10.1
RUN pip install ensemble-boxes

RUN pip install scikit-image==0.18
RUN pip install imagecorruptions
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda120


RUN apt update
RUN apt remove --purge --auto-remove cmake
RUN apt purge --auto-remove cmake
RUN apt update

RUN apt install -y software-properties-common lsb-release
RUN apt install -y libgl1-mesa-glx
RUN apt clean all



WORKDIR /workspace
