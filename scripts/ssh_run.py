"""SSH helper for the DAMODEL host: paramiko + scripted commands.

Usage:
    python ssh_run.py probe        # one-shot system probe
    python ssh_run.py 'free -m'    # arbitrary remote command
    python ssh_run.py upload <local> <remote>
    python ssh_run.py download <remote> <local>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import paramiko

HOST = "cn-north-b.ssh.damodel.com"
PORT = 38492
USER = "root"
PASSWORD = "nHWrJlvmhJ"


def _client() -> paramiko.SSHClient:
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(
        HOST,
        port=PORT,
        username=USER,
        password=PASSWORD,
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
        # Avoid agent / key file auth so paramiko goes straight to password.
        look_for_keys=False,
        allow_agent=False,
    )
    return cli


def run(cmd: str, *, timeout: float = 120) -> tuple[int, str, str]:
    with _client() as cli:
        stdin, stdout, stderr = cli.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        rc = stdout.channel.recv_exit_status()
    return rc, out, err


def upload(local: Path, remote: str) -> int:
    size = local.stat().st_size
    with _client() as cli, cli.open_sftp() as sftp:
        # mkdir -p the parent
        parent = "/".join(remote.split("/")[:-1])
        if parent:
            cli.exec_command(f'mkdir -p "{parent}"')[1].channel.recv_exit_status()
        last_pct = -1
        def progress(done: int, total: int) -> None:
            nonlocal last_pct
            pct = int(done / total * 100) if total else 0
            if pct != last_pct and pct % 5 == 0:
                print(f"  upload {pct}%  {done/(1024**2):.1f}/{total/(1024**2):.1f} MiB", flush=True)
                last_pct = pct
        sftp.put(str(local), remote, callback=progress)
    print(f"uploaded {local} -> {remote} ({size/(1024**2):.1f} MiB)")
    return 0


def download(remote: str, local: Path) -> int:
    local.parent.mkdir(parents=True, exist_ok=True)
    with _client() as cli, cli.open_sftp() as sftp:
        sftp.get(remote, str(local))
    print(f"downloaded {remote} -> {local}")
    return 0


def probe() -> int:
    cmds = [
        ("uname",            "uname -a"),
        ("os-release",       "cat /etc/os-release | head -5"),
        ("cpu",              "nproc; lscpu | head -10"),
        ("mem",              "free -h | head -3"),
        ("disk",             "df -h | grep -E '^(Filesystem|/dev|tmpfs.*\\s/)' | head -10"),
        ("home",             "echo HOME=$HOME; ls -la $HOME 2>/dev/null | head -20"),
        ("pwd",              "pwd"),
        ("python",           "which python3; python3 --version 2>&1; which python; python --version 2>&1"),
        ("pip",              "python3 -m pip --version 2>&1; python3 -m pip list 2>&1 | head -30"),
        ("git",              "git --version"),
        ("nvidia",           "nvidia-smi 2>&1 | head -25 || echo 'no nvidia-smi'"),
        ("cuda",             "ls /usr/local/cuda* 2>/dev/null; nvcc --version 2>&1 | head -5 || echo 'no nvcc'"),
        ("net",              "curl -sS --max-time 5 -o /dev/null -w 'github=%{http_code} %{time_total}s\\n' https://github.com/ 2>&1; "
                              "curl -sS --max-time 5 -o /dev/null -w 'huggingface=%{http_code} %{time_total}s\\n' https://huggingface.co/ 2>&1; "
                              "curl -sS --max-time 5 -o /dev/null -w 'hf-mirror=%{http_code} %{time_total}s\\n' https://hf-mirror.com/ 2>&1; "
                              "curl -sS --max-time 5 -o /dev/null -w 'modelscope=%{http_code} %{time_total}s\\n' https://modelscope.cn/ 2>&1"),
        ("hf-cli",           "which huggingface-cli; huggingface-cli --version 2>&1 | head -3"),
        ("workdir",          "ls -la /root /workspace 2>/dev/null"),
    ]
    fails: list[str] = []
    for label, cmd in cmds:
        print(f"\n=== {label} === {cmd}")
        rc, out, err = run(cmd, timeout=15)
        sys.stdout.write(out)
        if err.strip():
            sys.stdout.write("[stderr] " + err)
        if rc != 0:
            print(f"[rc={rc}]")
            fails.append(label)
    if fails:
        print(f"\nNON-ZERO EXITS: {fails}")
    return 0 if not fails else 1


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd = sys.argv[1]
    if cmd == "probe":
        return probe()
    if cmd == "upload":
        return upload(Path(sys.argv[2]), sys.argv[3])
    if cmd == "download":
        return download(sys.argv[2], Path(sys.argv[3]))
    # else: arbitrary remote command
    rc, out, err = run(" ".join(sys.argv[1:]), timeout=600)
    sys.stdout.write(out)
    if err.strip():
        sys.stderr.write(err)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
