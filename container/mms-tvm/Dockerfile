FROM nvcr.io/nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# Set a docker label to advertise multi-model support on the container
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
# Set a docker label to enable container to use SAGEMAKER_BIND_TO_PORT environment variable if present
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

# Install necessary dependencies for MMS and SageMaker Inference Toolkit
RUN apt-get update && \
    apt-get -y install --no-install-recommends \
    build-essential \
    ca-certificates \
    openjdk-8-jdk-headless \
    python3-dev \
    curl wget \
    vim lsof net-tools \
    python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev git \
    && rm -rf /var/lib/apt/lists/* \
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/local/bin/pip pip /usr/local/bin/pip3 1

# Install MXNet, MMS, and SageMaker Inference Toolkit to set up MMS
RUN pip3 --no-cache-dir install mxnet-cu102 \
                                multi-model-server \
                                sagemaker-inference \
                                retrying ujson orjson

RUN wget https://cmake.org/files/v3.17/cmake-3.17.2-Linux-x86_64.sh \
    && bash cmake-3.17.2-Linux-x86_64.sh --skip-license --prefix=/usr/local

COPY TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn7.6.tar.gz TensorRT.tar.gz

RUN tar xzvf TensorRT.tar.gz && cd /TensorRT-7.0.0.11/python && pip3 install tensorrt-7.0.0.11-cp36-none-linux_x86_64.whl \
    && cd ../graphsurgeon && pip3 install graphsurgeon-0.4.1-py2.py3-none-any.whl

ENV LD_LIBRARY_PATH=/TensorRT-7.0.0.11/lib:$LD_LIBRARY_PATH

RUN rm -rf TensorRT.tar.gz && git clone --recursive https://github.com/neo-ai/tvm tvm && cd tvm && git checkout release-1.3.0 \
    && mkdir build && cp cmake/config.cmake build && cd build \
    && sed -i 's/set(USE_TENSORRT OFF)/set(USE_TENSORRT \/TensorRT-7.0.0.11\/)/g' config.cmake && sed -i 's/set(USE_CUDA OFF)/set(USE_CUDA ON)/g' config.cmake \
    && cmake .. && make -j`nproc` && cd ../python; python3 setup.py install --user; cd ..

# Copy entrypoint script to the image
COPY dockerd-entrypoint.py /usr/local/bin/dockerd-entrypoint.py
RUN chmod +x /usr/local/bin/dockerd-entrypoint.py

RUN mkdir -p /home/model-server/

# Copy the default custom service file to handle incoming data and inference requests
COPY model_handler.py /home/model-server/model_handler.py

RUN echo "vmargs=-XX:-UseContainerSupport -Dlog4j.configuration=file:///log4j.properties" >> /usr/local/lib/python3.6/dist-packages/sagemaker_inference/etc/default-mms.properties

# Define an entrypoint script for the docker image
ENTRYPOINT ["python", "/usr/local/bin/dockerd-entrypoint.py"]

# Define command to be passed to the entrypoint
CMD ["serve"]

