$processName = "python"

Get-Process -Name $processName -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -like "*pythoncore-3.14-64*" -and $_.CommandLine -like "*uvicorn*" } |
    Stop-Process -Force
