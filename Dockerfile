# TRELLIS.2 (microsoft/TRELLIS.2-4B) HAZIR ÜRETİM İMAJI — 19.09
# Neden: 3 haftada 5 pod kurulumu ayrı ayrı çöktü (conda ToS/Py3.14, torch 2.14+cu130, flash-attn derlemesi,
# transformers sürümü). Kurulum bir kez BURADA yapılır, doğrulanır, dondurulur; pod açılınca üretim hemen başlar.
# Reçete = 19.09'da pod 51595671 üzerinde ÇALIŞTIĞI DOĞRULANAN adımlar (arac/t2-pod-kur-torch26.sh), conda'sız.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# 19.09 1. inşa: "runner lost communication" — 7 GB RAM koşucuda MAX_JOBS=4 nvcc belleği tüketti → MAX_JOBS=1 + 8 GB takas
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.6;8.9" \
    MAX_JOBS=1 \
    HF_HOME=/workspace/hf \
    PYTHONPATH=/workspace/T2 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Ubuntu 22.04 → python3.10 (TRELLIS.2 README'nin istediği sürüm; wheel'ler cp310)
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
      git wget curl ca-certificates openssh-server build-essential ninja-build \
      python3 python3-dev python3-pip python3-venv libgl1 libglib2.0-0 libx11-6 \
    && rm -rf /var/lib/apt/lists/* && mkdir -p /run/sshd /workspace

WORKDIR /workspace
RUN python3 -m venv /workspace/venv && /workspace/venv/bin/pip install -q --upgrade pip setuptools wheel
ENV PATH=/workspace/venv/bin:$PATH

# 1) torch 2.6.0 + cu124 — setup.sh'ın pin'i başka bağımlılıkla ezilmişti (2.14+cu130 gelip uzantılar derlenmedi);
#    burada ÖNCE ve sabit kurulur, sonra hiçbir adım torch'a dokunmaz.
RUN pip install -q torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# 2) TRELLIS.2 deposu (alt modüller ŞART: o_voxel/flexgemm orada) — commit sabit, "main" kaymasın
ARG T2_COMMIT=75fbf0183001
RUN git clone --recursive https://github.com/microsoft/TRELLIS.2.git /workspace/T2 \
    && cd /workspace/T2 && git checkout -q ${T2_COMMIT} && git submodule update --init --recursive -q

# 3) setup.sh --basic'in satırları (transformers SÜRÜMLÜ: 4.57.1 — 5.x DINOv3 API'sini kırıyor, 16.09 dersi)
RUN pip install -q imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh \
      'transformers==4.57.1' tensorboard pandas lpips zstandard kornia timm huggingface_hub \
    && pip install -q git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# 4) flash-attn: HAZIR WHEEL (kaynaktan derleme saatler sürüp 19.09'da bir turu yaktı). torch2.6 + cu12 + cp310 + cxx11abiFALSE
RUN pip install -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# 5) CUDA uzantıları — GPU OLMADAN derlenir (nvcc + TORCH_CUDA_ARCH_LIST yeter). 8.6=3090/A6000, 8.9=4090/L40
RUN git clone -q -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/ext/nvdiffrast \
    && pip install -q /tmp/ext/nvdiffrast --no-build-isolation \
    && git clone -q -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/ext/nvdiffrec \
    && pip install -q /tmp/ext/nvdiffrec --no-build-isolation \
    && rm -rf /tmp/ext
RUN git clone -q --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/ext/CuMesh \
    && pip install -q /tmp/ext/CuMesh --no-build-isolation && rm -rf /tmp/ext
RUN git clone -q --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/ext/FlexGEMM \
    && pip install -q /tmp/ext/FlexGEMM --no-build-isolation && rm -rf /tmp/ext
RUN pip install -q /workspace/T2/o-voxel --no-build-isolation

# 6) KURULUM KAPISI — eksik modül varsa imaj ÜRETİLMEZ (sessiz başarısızlık yok; 19.09'da "KURULUM BİTTİ" yazıp torch'suz kalmıştı)
RUN for M in torch flash_attn o_voxel cumesh flex_gemm nvdiffrast transformers trellis2 utils3d; do \
      python3 -c "import $M" || { echo "✗ EKSİK: $M"; exit 1; }; done \
    && python3 -c "import torch,transformers;print('torch',torch.__version__,'cuda',torch.version.cuda,'transformers',transformers.__version__)"

# 7) AĞIRLIKLAR imaja gömülür (~8 GB; pod'da 30 dk indirme + $0,30 trafik bedeli bitiyor). TRELLIS.2-4B MIT lisanslı, açık.
#    DINOv3 (facebook/dinov3-vitl16) GATED → gömülmez, çalışma anında HF_TOKEN ile iner (~1,2 GB).
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/TRELLIS.2-4B')" \
    && du -sh /workspace/hf

# 8) Üretim betikleri
COPY trellis2-toplu.py t2-basla.sh /workspace/
RUN chmod +x /workspace/t2-basla.sh && mkdir -p /workspace/girdi /workspace/cikti-toplu /workspace/isaret
# vast.ai onstart bu kabuğu bulur; SSH'ı vast kendi başlatır.
CMD ["/bin/bash"]
