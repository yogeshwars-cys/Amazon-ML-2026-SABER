#!/bin/bash
# Run on the laptop: bundle the small work files + encoder and send them with the challenge zip.
# usage: bash pod/upload.sh <ssh-port> <ip>
set -euo pipefail
PORT=$1; IP=$2; ROOT="D:/Downloads/projects/Amazon ML chal"; W="$ROOT/work"
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new root@$IP"
tar -cf "$W/work_bundle.tar" -C "$W" q.parquet pool.parquet partitions.parquet region_model_E.json \
    mini_source1.parquet mini_source2.parquet mini_source3.parquet ft_arctic-xs-mean_E3k
$SSH "mkdir -p /workspace/upload"
scp -P $PORT "$W/work_bundle.tar" root@$IP:/workspace/upload/work_bundle.tar
scp -P $PORT "$ROOT/6ab10eb3b23ba_student_resource.zip" root@$IP:/workspace/upload/student_resource.zip
echo uploaded
