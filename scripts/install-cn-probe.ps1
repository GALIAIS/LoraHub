# Probe per-endpoint candidate mirrors and emit the fastest as
# KEY=VALUE lines on stdout, ready for the .bat wrapper to consume.
# Diagnostics go to stderr.
$ErrorActionPreference = 'Stop'
$ProgressPreference   = 'SilentlyContinue'

function Probe-Url {
    param([string]$Url)
    try {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $req = [System.Net.HttpWebRequest]::Create($Url)
        $req.Method = 'GET'
        $req.AddRange(0, 262143)
        $req.Timeout = 5000
        $req.ReadWriteTimeout = 5000
        $req.AllowAutoRedirect = $true
        $resp = $req.GetResponse()
        $stream = $resp.GetResponseStream()
        $buf = New-Object byte[] 8192
        while ($stream.Read($buf, 0, $buf.Length) -gt 0) {}
        $stream.Close()
        $resp.Close()
        $sw.Stop()
        return $sw.ElapsedMilliseconds
    } catch {
        return -1
    }
}

function Pick-Fastest {
    param(
        [string]$Label,
        [string[]]$Candidates,
        [scriptblock]$ProbeUrl
    )
    [Console]::Error.WriteLine("  probing ${Label} ...")
    $best = $null
    $bestMs = [int]::MaxValue
    $haveAny = $false
    foreach ($cand in $Candidates) {
        $url = & $ProbeUrl $cand
        $ms = Probe-Url -Url $url
        $display = if ([string]::IsNullOrEmpty($cand)) { '(direct)' } else { $cand }
        if ($ms -ge 0) {
            [Console]::Error.WriteLine(("    {0,4}ms  {1}" -f $ms, $display))
            $haveAny = $true
            if ($ms -lt $bestMs) {
                $bestMs = $ms
                $best = $cand
            }
        } else {
            [Console]::Error.WriteLine("    fail    $display")
        }
    }
    if (-not $haveAny) {
        $best = $Candidates[0]
        $fallback = if ([string]::IsNullOrEmpty($best)) { '(direct)' } else { $best }
        [Console]::Error.WriteLine("  [!] all probes failed; using $fallback")
    }
    return $best
}

# --- Candidate pools ----------------------------------------------------

$ghProxies = @(
  ''  # empty = direct GitHub, in case network is unrestricted
  'https://v4.gh-proxy.org/'
  'https://gh-proxy.com/'
  'https://gh.ddlc.top/'
  'https://gh.jasonzeng.dev/'
  'https://gh.zwy.one/'
  'https://gh-proxy.org/'
  'https://hk.gh-proxy.org/'
  'https://v6.gh-proxy.org/'
  'https://ghfast.top/'
)

$pythonBuildMirrors = @(
  'https://registry.npmmirror.com/-/binary/python-build-standalone'
  'https://github.com/astral-sh/python-build-standalone/releases/download'
)

$pypiIndexes = @(
  'https://mirrors.aliyun.com/pypi/simple'
  'https://mirror.nju.edu.cn/pypi/web/simple'
  'https://mirrors.bfsu.edu.cn/pypi/web/simple'
  'https://pypi.mirrors.ustc.edu.cn/simple'
  'https://pypi.tuna.tsinghua.edu.cn/simple'
  'https://mirrors.cloud.tencent.com/pypi/simple'
  'https://mirrors.huaweicloud.com/repository/pypi/simple'
  'https://pypi.org/simple'
)

$nodeMirrors = @(
  'https://npmmirror.com/mirrors/node'
  'https://mirrors.tuna.tsinghua.edu.cn/nodejs-release'
  'https://mirrors.aliyun.com/nodejs-release'
  'https://nodejs.org/dist'
)

$npmRegistries = @(
  'https://registry.npmmirror.com'
  'https://registry.npmjs.org'
)

$pytorchWheelMirrors = @(
  'https://mirrors.aliyun.com/pytorch-wheels'
  'https://mirrors.nju.edu.cn/pytorch/whl'
  'https://download.pytorch.org/whl'
  'https://mirror.sjtu.edu.cn/pytorch-wheels'
)

# --- Run probes ---------------------------------------------------------

$gh   = Pick-Fastest 'GitHub proxy'           $ghProxies          { param($c) "${c}https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip" }
$py   = Pick-Fastest 'python-build-standalone' $pythonBuildMirrors { param($c) if ($c -match 'npmmirror') { "$c/" } else { $c } }
$pypi = Pick-Fastest 'PyPI'                   $pypiIndexes        { param($c) "$c/pip/" }
$node = Pick-Fastest 'Node binary'            $nodeMirrors        { param($c) "$c/index.json" }
$npm  = Pick-Fastest 'npm registry'           $npmRegistries      { param($c) "$c/lodash" }
$torch = Pick-Fastest 'PyTorch wheel'         $pytorchWheelMirrors { param($c) "$c/cu128/torch/" }

# --- Emit KEY=VALUE for the .bat wrapper -------------------------------

Write-Output "LORAHUB_GH_PROXY=$gh"
Write-Output "UV_PYTHON_INSTALL_MIRROR=$py"
Write-Output "UV_INDEX_URL=$pypi"
Write-Output "UV_DEFAULT_INDEX=$pypi"
Write-Output "LORAHUB_NODE_MIRROR=$node"
Write-Output "NPM_CONFIG_REGISTRY=$npm"
Write-Output "LORAHUB_TORCH_INDEX_URL=$torch"
