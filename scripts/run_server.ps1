$pythonExe = "C:/Users/conne/AppData/Local/Python/pythoncore-3.14-64/python.exe"
Set-Location $PSScriptRoot/..
& $pythonExe -m uvicorn app.main:app --reload
