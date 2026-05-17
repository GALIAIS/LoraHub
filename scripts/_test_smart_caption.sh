#!/usr/bin/env bash
# Quick test of the smart-caption endpoint on VPS
curl -s -w '\nHTTP_STATUS: %{http_code}\n' \
  http://127.0.0.1:6006/api/image-studio/ai/smart-caption \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"path":"/root/autodl-tmp/LoraHub/datasets/lx","recursive":false}' \
  | head -50
