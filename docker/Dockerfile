# FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
FROM pytorch/pytorch:latest

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV BENCHMARK_REPO https://github.com/greenstar1151/pytorch-benchmark

# RUN apt-get update
# RUN apt-get install gcc -y
# RUN pip3 uninstall torch torchvision torchtext -y
# RUN pip3 install --pre torch torchvision torchtext -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html
# RUN /opt/conda/bin/conda clean -tipsy && \
#     ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
#     echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
#     echo "conda activate base" >> ~/.bashrc

# RUN /opt/conda/bin/conda uninstall --force pytorch torchvision torchtext
# RUN /opt/conda/bin/conda install -c pytorch-nightly pytorch torchvision torchtext

# Need to add as per https://stackoverflow.com/questions/55313610
RUN apt-get update
RUN apt-get install git jq ffmpeg libsm6 libxext6 g++ -y

RUN git clone -b power_nsightsystems --single-branch ${BENCHMARK_REPO} /workspace/benchmark
RUN cd /workspace/benchmark;
RUN python install.py
RUN mkdir /workspace/benchmark/ncu_reports
RUN  echo "deb [trusted=yes] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list
RUN apt-get --allow-unauthenticated update
RUN apt-get --allow-unauthenticated install nsight-systems-2021.2.4
RUN pip install -e git+https://github.com/fbcotter/py3nvml#egg=py3nvml
#ENV PATH="/opt/nvidia/nsight-compute/2021.2.0/:$PATH"
#RUN dpkg -L nsight-compute-2021.2.0
# --continue_on_fail
# RUN pip3 install --upgrade numpy==1.17.3
