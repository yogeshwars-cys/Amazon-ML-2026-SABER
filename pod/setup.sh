#!/bin/bash
# One-time pod setup. Expects /workspace/upload/{student_resource.zip,work_bundle.tar} (sent by pod/upload.sh).
set -euo pipefail
cd /workspace
[ -d saber ] || git clone -q https://github.com/yogeshwars-cys/Amazon-ML-2026-SABER saber
cd saber && git pull -q
grep -vE '^torch' requirements.txt > /tmp/req.txt                     # keep the template's CUDA torch
pip install -q -r /tmp/req.txt
command -v unzip >/dev/null || (apt-get update -qq && apt-get install -y -qq unzip >/dev/null)
mkdir -p /workspace/data /workspace/work
[ -d /workspace/data/student_resource ] || unzip -q /workspace/upload/student_resource.zip -x '__MACOSX/*' -d /workspace/data
tar -xf /workspace/upload/work_bundle.tar -C /workspace/work
python -c "import torch, polars, lightgbm, rapidfuzz, sentence_transformers as st; print('torch', torch.__version__, torch.cuda.get_device_name(0), 'st', st.__version__)"
echo "RAM GB: $(free -g | awk '/Mem:/ {print $2}')  vCPU: $(nproc)  disk: $(df -h /workspace | awk 'NR==2 {print $4}') free"
