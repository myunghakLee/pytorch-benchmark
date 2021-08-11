# FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
FROM pytorch/pytorch:latest

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8


# Need to add as per https://stackoverflow.com/questions/55313610
RUN  echo "deb [trusted=yes] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list
RUN apt-get --allow-unauthenticated update
RUN apt-get install git jq ffmpeg libsm6 libxext6 g++ -y
RUN apt-get --allow-unauthenticated install nsight-systems-2021.2.4

COPY . /workspace/benchmark
WORKDIR /workspace/benchmark
RUN mkdir /trace_data
RUN mkdir /trace_data/exec_time
RUN mkdir /trace_data/ncu_reports

RUN pip3 install -e git+https://github.com/fbcotter/py3nvml#egg=py3nvml

RUN python3 ./install.py
RUN rm -rf /var/lib/apt/lists/*
RUN conda install notebook -y

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
