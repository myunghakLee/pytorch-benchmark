# FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
FROM pytorch/pytorch:latest

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN  echo "deb [trusted=yes] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list
RUN apt-get --allow-unauthenticated update
RUN apt-get install git jq ffmpeg libsm6 libxext6 g++ -y
RUN apt-get --allow-unauthenticated install nsight-systems-2021.2.4
RUN pip3 install -e git+https://github.com/fbcotter/py3nvml#egg=py3nvml

COPY . /workspace/benchmark
WORKDIR /workspace/benchmark
RUN mkdir /ncu_reports
RUN python3 ./install.py
RUN rm -rf /var/lib/apt/lists/*