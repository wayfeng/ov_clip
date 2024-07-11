FROM ubuntu:22.04 AS builder
RUN apt update && apt install -y  --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    python3-wheel build-essential && \
    apt clean && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /home/gradio/venv
ENV PATH="/home/gradio/venv/bin:$PATH"
COPY requirements_app.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

FROM ubuntu:22.04 AS runner
RUN apt update && apt install --no-install-recommends -y \
    python3.10 python3-venv curl gnupg ca-certificates && \
    apt clean && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home gradio
COPY --from=builder /home/gradio/venv /home/gradio/venv

RUN curl https://repositories.intel.com/gpu/intel-graphics.key | \
    gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg && \
    echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client" | \
    tee /etc/apt/sources.list.d/intel-gpu-jammy.list && \
    apt update && \
    apt install --no-install-recommends -y \
    intel-opencl-icd intel-level-zero-gpu level-zero \
    intel-media-va-driver-non-free libmfx1 libmfxgen1 libvpl2 \
    libegl-mesa0 libegl1-mesa libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri \
    libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm12 libxatracker2 mesa-va-drivers \
    mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all vainfo hwinfo clinfo && \
    apt clean && rm -rf /var/lib/apt/lists/*
#USER gradio
RUN mkdir -p /home/gradio/app
WORKDIR /home/gradio/app
COPY app.py tokenizer.py utils.py bpe_simple_vocab_16e6.txt.gz ./
COPY config_docker.py ./config.py
#COPY models models
EXPOSE 7580
ENV PYTHONUNBUFFERED=1
ENV VIRTUAL_ENV=/home/gradio/venv
ENV PATH="/home/gradio/venv/bin:$PATH"
#CMD ["python","app.py"]
