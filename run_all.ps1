param(
    [string]$PythonExe = "",
    [switch]$CheckReproducibility
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if (-not $PythonExe) {
    $PythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $PythonExe)) {
        throw "缺少 .venv。请先运行 .\setup_environment.ps1，或用 -PythonExe 指定已有环境的解释器。"
    }
}
foreach ($dataName in @("train_data.csv", "test_data_unlabeled.csv")) {
    if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot "data\raw\$dataName"))) {
        throw "缺少教师数据 data\raw\$dataName。请确认仓库文件已完整下载。"
    }
}

function Invoke-LabPython([string[]]$Arguments) {
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python 命令执行失败（退出码 $LASTEXITCODE）：$Arguments"
    }
}

$env:PYTHONHASHSEED = "42"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"

Invoke-LabPython -Arguments @("-m", "unittest", "discover", "-s", "tests", "-v")
Invoke-LabPython -Arguments @("-m", "src.run_experiments", "--config", "configs\experiment_config.json")
Invoke-LabPython -Arguments @("-m", "src.verify_project")
if ($CheckReproducibility) {
    Invoke-LabPython -Arguments @("-m", "src.reproducibility_check", "--save-baseline")
    Invoke-LabPython -Arguments @("-m", "src.run_experiments", "--config", "configs\experiment_config.json")
    Invoke-LabPython -Arguments @("-m", "src.verify_project")
    Invoke-LabPython -Arguments @("-m", "src.reproducibility_check")
}
& $PythonExe -m pip freeze | Set-Content -Encoding utf8 requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw "导出环境依赖失败。" }

Write-Host "实验、预测与完整性检查均已完成。"
