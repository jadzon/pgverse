# Bazowy obraz z CUDA 12.1.1 i cuDNN8 na Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Ustawienia non-interactive dla APT
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Warsaw

# 1) Systemowe zależności + Python 3.10
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      tzdata \
      build-essential \
      poppler-utils \
      tesseract-ocr tesseract-ocr-pol \
      libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
      python3.10 python3-pip python3.10-dev python3-tk \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

# 2) Symlink python3.10 -> python i update pip
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    pip install --upgrade pip

WORKDIR /app

# 3) GPU-owy PyTorch + torchvision + torchaudio (CUDA 12.1)
RUN pip install --no-cache-dir \
      torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu121

# 4) Pythonowe biblioteki
COPY requirements.txt ./
RUN sed -i '/^onnxruntime/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir onnxruntime-gpu einops timm paddleocr paddlepaddle-gpu tokenizers==0.21.0 img2table

# 5) Detectron2: budowa ze źródeł dla CUDA 12.1
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git cmake ninja-build && \
    rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/facebookresearch/detectron2.git /detectron2 && \
    python -m pip install --no-cache-dir /detectron2 && \
    rm -rf /detectron2

# 6) LayoutParser z obsługą detectron2
RUN pip install --no-cache-dir layoutparser[detectron2]

# 7) Kopiujemy cały kod aplikacji
COPY . .

# 8) Headless matplotlib
ENV MPLBACKEND=Agg

# 9) Domyślne polecenie startowe
CMD ["python", "-u", "mozg_ocr.py"]