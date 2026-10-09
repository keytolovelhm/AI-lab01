param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

try {
    & $PythonExe --version
} catch {
    throw "未找到 Python。请安装 Python 3.12，或使用 -PythonExe 传入 python.exe 的完整路径。"
}
if ($LASTEXITCODE -ne 0) { throw "Python 版本检查失败。" }
& $PythonExe -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)"
if ($LASTEXITCODE -ne 0) { throw "固定依赖需要 Python 3.12 或更新版本；建议使用实验原版本 Python 3.12。" }

& $PythonExe -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "创建虚拟环境失败。" }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "更新 pip 失败。" }
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "安装实验依赖失败。" }
& .\.venv\Scripts\python.exe -m pip freeze | Set-Content -Encoding utf8 requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw "导出环境依赖失败。" }
Write-Host "环境已创建：$PSScriptRoot\.venv"
