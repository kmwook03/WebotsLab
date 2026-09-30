param(
    [string]$WebotsExecutable = $env:WEBOTS_EXECUTABLE
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$world = Join-Path $repoRoot 'worlds\amr_search_rescue.wbt'

if (-not $WebotsExecutable) {
    $candidates = @(
        'C:\Program Files\Webots\msys64\mingw64\bin\webots.exe',
        'C:\Program Files\Webots\webots.exe'
    )
    $WebotsExecutable = $candidates |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
        Select-Object -First 1
}
if (-not $WebotsExecutable -or -not (Test-Path -LiteralPath $WebotsExecutable -PathType Leaf)) {
    throw 'Webots executable not found. Set WEBOTS_EXECUTABLE to its absolute path.'
}

$resultPath = Join-Path ([System.IO.Path]::GetTempPath()) (
    'webots-amr-evaluation-' + [guid]::NewGuid().ToString('N') + '.json'
)
$previousResultPath = $env:AMR_EVALUATION_RESULT_PATH
try {
    $env:AMR_EVALUATION_RESULT_PATH = $resultPath
    # Do not pipe Webots output on Windows. Webots' Qt launcher can derive an
    # invalid window size when its standard handles are redirected.
    & $WebotsExecutable `
        --batch `
        --mode=fast `
        --no-rendering `
        --stdout `
        --stderr `
        $world
    $webotsExitCode = $LASTEXITCODE

    if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
        throw "Evaluator produced no result file (Webots exit code $webotsExitCode)."
    }
    $payload = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
    if ($payload.outcome -ne 'PASS') {
        throw "Mission outcome was $($payload.outcome): $($payload.reason)"
    }
    if ($payload.phase -ne 'COMPLETE') {
        throw "PASS result reported an invalid controller phase: $($payload.phase)"
    }
    if ([int]$payload.targets_reached -ne [int]$payload.targets_required) {
        throw "Only $($payload.targets_reached)/$($payload.targets_required) targets were reached."
    }
    if ([double]$payload.elapsed_s -gt [double]$payload.time_limit_s) {
        throw "Mission exceeded the simulation deadline."
    }
    if ([double]$payload.home_distance_m -ge [double]$payload.home_limit_m) {
        throw "Robot finished outside the allowed home radius."
    }
    if ([double]$payload.closest_person_m -lt [double]$payload.person_limit_m) {
        throw "Robot violated the moving-person safety distance."
    }
    if ($webotsExitCode -ne 0) {
        throw "Webots returned exit code $webotsExitCode despite evaluator PASS."
    }

    Write-Host ($payload | ConvertTo-Json -Compress)
    Write-Host 'Webots end-to-end regression: PASS'
} finally {
    $env:AMR_EVALUATION_RESULT_PATH = $previousResultPath
    if (Test-Path -LiteralPath $resultPath) {
        Remove-Item -LiteralPath $resultPath -Force
    }
}
