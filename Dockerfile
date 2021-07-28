FROM nikolaik/python-nodejs:python3.9-nodejs16

RUN apt-get update && apt-get install -y \
  build-essential \
  wget \
  lsb-release \
  software-properties-common \
  python3 \
  python3-pip \
  python3-dev \
  python3-numpy \
  apt-transport-https \
  ca-certificates \
  libz-dev

RUN wget https://apt.llvm.org/llvm.sh && \
  chmod +x llvm.sh && \
  ./llvm.sh 11

RUN wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0-linux-x86_64.sh && \
    chmod +x cmake-3.20.0-linux-x86_64.sh && \
    mkdir /opt/cmake && \
    ./cmake-3.20.0-linux-x86_64.sh --skip-license --prefix=/opt/cmake && \
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake

RUN apt-get update \
  && apt-get install -y python3 python3-pip git jq \
       autoconf automake bison build-essential clang \
       doxygen flex g++ git libffi-dev libncurses5-dev \
       libtool libsqlite3-dev make mcpp python \
       sqlite zlib1g-dev htop bc parallel time

WORKDIR /arrow

RUN pip3 install xxhash scipy numpy && \
  git clone https://github.com/apache/arrow /arrow && \
  mkdir -p /build/arrow && \
  cd /build/arrow && \
  cmake /arrow/cpp \
    -DARROW_BUILD_SHARED=ON \
    -DARROW_BUILD_STATIC=ON \
    -DARROW_COMPUTE=ON \
    -DARROW_CSV=ON \
    -DARROW_DATASET=ON \
    -DARROW_DEPENDENCY_SOURCE=BUNDLED \
    -DARROW_DEPENDENCY_USE_SHARED=ON \
    -DARROW_FILESYSTEM=ON \
    -DARROW_HDFS=ON \
    -DARROW_JEMALLOC=ON \
    -DARROW_JSON=ON \
    -DARROW_ORC=ON \
    -DARROW_PYTHON=ON \
    -DARROW_PARQUET=ON \
    -DARROW_PLASMA=ON \
    -DARROW_WITH_BROTLI=ON \
    -DARROW_WITH_BZ2=ON \
    -DARROW_WITH_LZ4=ON \
    -DARROW_WITH_SNAPPY=ON \
    -DARROW_WITH_ZLIB=ON \
    -DARROW_WITH_ZSTD=ON \
    -DORC_SOURCE=BUNDLED && \
  make -j$(nproc) && \
  make install && \
  cd /arrow/python && \
  pip3 install cython pandas && \
  pip3 install -r requirements-build.txt && \
  PYARROW_WITH_PARQUET=1 PYARROW_WITH_DATASET=on python3 setup.py build_ext --inplace

ENV PATH=${PATH}:/scripts

ENV LD_LIBRARY_PATH=/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

RUN echo "fs.inotify.max_user_instances=524288" >> /etc/sysctl.conf
RUN echo "fs.inotify.max_user_watches=524288" >> /etc/sysctl.conf
RUN echo "fs.inotify.max_queued_events=524288" >> /etc/sysctl.conf

COPY . /app

WORKDIR /app/applications/jupyter-extension

RUN pip3 install -e . && pip3 install tqdm pandas numpy scipy nbconvert

ENTRYPOINT [ "bash" ]
