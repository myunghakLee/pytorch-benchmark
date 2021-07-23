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

RUN git clone -b master --single-branch ${BENCHMARK_REPO} /workspace/benchmark
RUN cd /workspace/benchmark; python install.py
# --continue_on_fail
# RUN pip3 install --upgrade numpy==1.17.3
RUN conda install notebook -y

WORKDIR /workspace/benchmark
# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]