# FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
FROM pytorch/pytorch:latest

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV BENCHMARK_REPO https://github.com/greenstar1151/pytorch-benchmark

# Need to add as per https://stackoverflow.com/questions/55313610
RUN apt-get update
RUN apt-get install git jq ffmpeg libsm6 libxext6 g++ -y

RUN git clone -b master --single-branch ${BENCHMARK_REPO} /workspace/benchmark
RUN cd /workspace/benchmark; python install.py
