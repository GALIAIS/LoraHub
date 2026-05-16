#!/usr/bin/env bash
# Run inside the remote shell once via sshpass+ssh.
set -e
echo "=== uname ==="; uname -a
echo "=== os ==="; cat /etc/os-release | head -4
echo "=== cpu/mem ==="; nproc; free -h | head -3
echo "=== disk ==="; df -h | grep -E '^(Filesystem|/dev|tmpfs.*\s/$)' | head -10
echo "=== home ==="; echo HOME=$HOME; ls -la $HOME 2>/dev/null | head -15
echo "=== gpu ==="; nvidia-smi 2>&1 | head -25 || echo "no nvidia-smi"
echo "=== cuda ==="; ls /usr/local/cuda* 2>/dev/null; nvcc --version 2>&1 | head -3 || echo "no nvcc"
echo "=== python ==="; for py in python python3 python3.10 python3.11 python3.12; do
  if command -v $py >/dev/null 2>&1; then echo "$py = $(command -v $py): $($py --version 2>&1)"; fi
done
echo "=== pip ==="; python3 -m pip --version 2>&1 | head -1
echo "=== git node npm uv ==="; git --version; node --version 2>/dev/null || echo "no node"; npm --version 2>/dev/null || echo "no npm"; uv --version 2>/dev/null || echo "no uv"
echo "=== net ==="
for url in https://github.com/ https://huggingface.co/ https://hf-mirror.com/ https://modelscope.cn/; do
  curl -sS --max-time 5 -o /dev/null -w "$(printf '%-32s' $url) http=%{http_code} time=%{time_total}s\n" $url 2>&1 || echo "    ($url unreachable)"
done
echo "=== hf-cli ==="; which huggingface-cli 2>&1; huggingface-cli --version 2>&1 | head -1
echo "=== /workspace etc ==="; ls -la / 2>/dev/null | head -25
echo "=== existing pip pkgs (pyt/diff/safe) ==="; python3 -m pip list 2>&1 | grep -iE '^(torch|diffusers|safetensors|transformers|accelerate|peft|huggingface)' || echo "(none of those installed)"
