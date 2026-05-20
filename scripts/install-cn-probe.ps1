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
        $req.Method = 'HEAD'
        $req.Timeout = 5000
        $req.ReadWriteTimeout = 5000
        $req.AllowAutoRedirect = $true
        $resp = $req.GetResponse()
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
        [Console]::Error.WriteLine("  [!] all probes failed, falling back to $fallback")
    }
    return $best
}

# --- Candidate pools ----------------------------------------------------

$ghProxies = @(
  ''  # empty = direct GitHub, in case network is unrestricted
  'https://gh-proxy.org/'
  'https://hk.gh-proxy.org/'
  'https://cdn.gh-proxy.org/'
  'https://v6.gh-proxy.org/'
  'https://ghfast.top/'
)

$pythonBuildMirrors = @(
  'https://registry.npmmirror.com/-/binary/python-build-standalone'
  'https://github.com/astral-sh/python-build-standalone/releases/download'
)

$pypiIndexes = @(
  'https://pypi.tuna.tsinghua.edu.cn/simple'
  'https://mirrors.aliyun.com/pypi/simple'
  'https://mirrors.cloud.tencent.com/pypi/simple'
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

# --- Run probes ---------------------------------------------------------

$gh   = Pick-Fastest 'GitHub proxy'           $ghProxies          { param($c) "${c}https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip" }
$py   = Pick-Fastest 'python-build-standalone' $pythonBuildMirrors { param($c) if ($c -match 'npmmirror') { "$c/" } else { $c } }
$pypi = Pick-Fastest 'PyPI'                   $pypiIndexes        { param($c) "$c/pip/" }
$node = Pick-Fastest 'Node binary'            $nodeMirrors        { param($c) "$c/index.json" }
$npm  = Pick-Fastest 'npm registry'           $npmRegistries      { param($c) "$c/lodash" }

# --- Emit KEY=VALUE for the .bat wrapper -------------------------------

Write-Output "LORAHUB_GH_PROXY=$gh"
Write-Output "UV_PYTHON_INSTALL_MIRROR=$py"
Write-Output "UV_INDEX_URL=$pypi"
Write-Output "LORAHUB_NODE_MIRROR=$node"
Write-Output "NPM_CONFIG_REGISTRY=$npm"
