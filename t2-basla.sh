#!/bin/bash
# t2-basla.sh — İMAJ SÜRÜMÜ. Pod açıldığında her şey kurulu; bu betik yalnız üretimi başlatır.
# Girdi: /workspace/girdi/*.png (RGBA, alpha maskeli — trellis2-toplu.py beyaz→alpha'yı kendi yapar)
# Çıktı: /workspace/cikti-toplu/<pn>.glb · günlük /workspace/uretim.log
# HF_TOKEN ortamda olmalı (DINOv3 gated). vast.ai: env alanına -e HF_TOKEN=... yazılır.
set -u
cd /workspace
export HF_HOME=/workspace/hf PYTHONPATH=/workspace/T2 PATH=/workspace/venv/bin:$PATH
[ -n "${HF_TOKEN:-}" ] || { echo "✗ HF_TOKEN yok — DINOv3 inmez"; exit 1; }
mkdir -p /workspace/cikti-toplu
echo "== $(date) ÜRETİM BAŞLIYOR · girdi: $(ls /workspace/girdi/*.png 2>/dev/null | wc -l) · GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
# 16.09 TUZAK 3 — 'nohup … &' ssh kapanınca düşüyordu; setsid + stdin /dev/null ŞART
setsid nohup /workspace/venv/bin/python3 -u /workspace/trellis2-toplu.py \
  --girdi=/workspace/girdi --cikti=/workspace/cikti-toplu \
  > /workspace/uretim.log 2>&1 < /dev/null &
sleep 3
pgrep -f trellis2-toplu.py >/dev/null && echo "ÜRETİM ÇALIŞIYOR pid=$(pgrep -f trellis2-toplu.py | head -1)" || { echo "✗ ÜRETİM BAŞLAMADI"; tail -20 /workspace/uretim.log; exit 1; }
