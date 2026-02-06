param(
  [string]$Python = "python"
)

function Resolve-PythonPath {
  param([string]$Cmd)
  try {
    $resolved = (Get-Command $Cmd -ErrorAction Stop).Source
    if ($resolved -like "*WindowsApps*") {
      throw "WindowsApps alias"
    }
    return $resolved
  } catch {
    $localPythonRoot = Join-Path $env:LOCALAPPDATA "Programs\Python"
    if (Test-Path $localPythonRoot) {
      $candidates = Get-ChildItem -Directory $localPythonRoot | Sort-Object Name -Descending
      foreach ($dir in $candidates) {
        $candidateExe = Join-Path $dir.FullName "python.exe"
        if (Test-Path $candidateExe) {
          return $candidateExe
        }
      }
    }
  }
  return $Cmd
}

$Python = Resolve-PythonPath -Cmd $Python

Write-Host "Creating venv (.venv)..."
& $Python -m venv .venv

Write-Host "Upgrading pip..."
& .\.venv\Scripts\python -m pip install -U pip

if (Test-Path "requirements.txt") {
  Write-Host "Installing requirements..."
  & .\.venv\Scripts\python -m pip install -r requirements.txt
} else {
  Write-Host "requirements.txt not found. Nothing to install."
}

Write-Host "Python version:"
& .\.venv\Scripts\python --version

Write-Host "Installed packages (top-level):"
& .\.venv\Scripts\python -m pip list
