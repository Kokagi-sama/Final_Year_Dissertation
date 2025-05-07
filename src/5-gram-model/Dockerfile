FROM ubuntu:20.04

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install essential tools and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libboost-all-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Flashlight-text
RUN pip3 install flashlight-text

# Clone and build KenLM
WORKDIR /opt
RUN git clone https://github.com/kpu/kenlm.git && \
    cd kenlm && \
    mkdir -p build && \
    cd build && \
    cmake .. && \
    make -j4

# Add KenLM to path
ENV PATH="/opt/kenlm/build/bin:${PATH}"

WORKDIR /data
COPY train_lm.sh /train_lm.sh
RUN chmod +x /train_lm.sh

ENTRYPOINT ["/train_lm.sh"]
