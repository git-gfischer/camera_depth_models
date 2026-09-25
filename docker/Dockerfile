# CDM on TensorRT: build engines, run on datasets, run live on a RealSense.
#
# Built and run through docker/docker-compose.yaml
# (`docker compose -f docker/docker-compose.yaml build engine`), or
#     docker build -f docker/Dockerfile -t cdm-trt:latest .
#
# Everything the image needs at run time is mounted, not baked in:
#     /config   config/ of the repository (engine, dataset, realsense settings)
#     /models   checkpoints, ONNX files, engines, parity references
#     /data     datasets to read
#     /output   results
# A TensorRT engine is tied to the TensorRT version and GPU that built it, so
# engines are built in a running container (`docker compose run --rm engine`),
# never here.

# CUDA 12.8 is the first release with Blackwell (sm_120, RTX 50xx) support.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

#   libopencv-dev                          C++ OpenCV with highgui for the live window
#   libyaml-cpp-dev                        the tools' config files
#   libusb/libudev/libssl                  librealsense
#   python3-venv                           the checkpoint -> ONNX export
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake curl git ninja-build pkg-config \
      libopencv-dev libyaml-cpp-dev \
      libusb-1.0-0-dev libudev-dev libssl-dev udev \
      python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# TensorRT, from the NVIDIA apt repository the CUDA image already carries.
# libnvinfer-bin provides trtexec, which builds the engines.
# ---------------------------------------------------------------------------
ARG TENSORRT_PACKAGES="libnvinfer-bin libnvinfer-dev"
RUN apt-get update && apt-get install -y --no-install-recommends ${TENSORRT_PACKAGES} \
    && rm -rf /var/lib/apt/lists/* \
    && grep -h -E "define (NV_TENSORRT|TRT)_(MAJOR|MINOR|PATCH)" /usr/include/x86_64-linux-gnu/NvInferVersion.h \
    && command -v trtexec

# ---------------------------------------------------------------------------
# librealsense, from source with the libusb (RSUSB) backend: it talks to the
# camera over USB from user space, so it works in a container without the
# patched uvcvideo kernel module.
# ---------------------------------------------------------------------------
ARG LIBREALSENSE_VERSION=2.58.4
RUN git clone --depth 1 --branch v${LIBREALSENSE_VERSION} \
      https://github.com/IntelRealSense/librealsense.git /tmp/librealsense \
    && cmake -S /tmp/librealsense -B /tmp/librealsense/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DFORCE_RSUSB_BACKEND=ON \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_GRAPHICAL_EXAMPLES=OFF \
      -DBUILD_TOOLS=ON \
      -DBUILD_PYTHON_BINDINGS=OFF \
      -DBUILD_UNIT_TESTS=OFF \
      -DCHECK_FOR_UPDATES=OFF \
    && cmake --build /tmp/librealsense/build --target install \
    && ldconfig \
    && rm -rf /tmp/librealsense

# ---------------------------------------------------------------------------
# Python for the export only (checkpoint -> ONNX + parity reference). CUDA
# wheels: the fp16 export traces the network in half precision, which needs
# GPU kernels.
# ---------------------------------------------------------------------------
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ENV CDM_PY=/opt/cdm-py/bin/python
RUN python3 -m venv /opt/cdm-py \
    && /opt/cdm-py/bin/pip install --no-cache-dir \
         torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
         --index-url https://download.pytorch.org/whl/cu128 \
    && /opt/cdm-py/bin/pip install --no-cache-dir numpy onnx opencv-python-headless pyyaml \
    && ${CDM_PY} -c "import torch, torchvision, onnx, cv2, yaml; print('torch', torch.__version__, 'onnx', onnx.__version__)"

# ---------------------------------------------------------------------------
# This repository: the Python model (for the export) and the C++ tools.
# ---------------------------------------------------------------------------
ENV CDM_ROOT=/opt/cdm
COPY camera_depth_models ${CDM_ROOT}/camera_depth_models
COPY scripts ${CDM_ROOT}/scripts
COPY assets/example_data ${CDM_ROOT}/assets/example_data
COPY docker/build_engine.sh /usr/local/bin/cdm_build_engine
COPY cpp ${CDM_ROOT}/cpp
# COPY keeps the checkout's modes, and a umask of 077 leaves files unreadable
# to the non-root user the compose services run as.
RUN chmod -R a+rX ${CDM_ROOT} && chmod 755 /usr/local/bin/cdm_build_engine
RUN cmake -S ${CDM_ROOT}/cpp -B ${CDM_ROOT}/build -G Ninja \
    && cmake --build ${CDM_ROOT}/build \
    && ctest --test-dir ${CDM_ROOT}/build --output-on-failure \
    && cmake --install ${CDM_ROOT}/build \
    && command -v cdm_dataset cdm_probe cdm_realsense

# Written by any user: the compose services run as the caller's UID so that
# outputs belong to them.
RUN mkdir -p /config /models /data /output && chmod 777 /models /output
# That UID has no passwd entry here; getpass.getuser() (PyTorch's cache paths)
# then falls back to these.
ENV HOME=/tmp USER=cdm LOGNAME=cdm
WORKDIR /output
CMD ["bash"]
