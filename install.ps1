# ghostframe installer for Windows:
#   irm https://raw.githubusercontent.com/Lyamon4/ghostframe/main/install.ps1 | iex
$ErrorActionPreference = 'Stop'
$repo = 'Lyamon4/ghostframe'
$dir = Join-Path $env:LOCALAPPDATA 'ghostframe\bin'
New-Item -ItemType Directory -Force $dir | Out-Null

$zip = Join-Path $env:TEMP 'ghostframe-windows-x64.zip'
Write-Host "Downloading the latest ghostframe release..."
if ($env:GHOSTFRAME_ZIP) { Copy-Item $env:GHOSTFRAME_ZIP $zip }   # local build, for testing
else { Invoke-WebRequest "https://github.com/$repo/releases/latest/download/ghostframe-windows-x64.zip" -OutFile $zip }
Stop-Process -Name ghostframe-tray -ErrorAction SilentlyContinue   # an older version may hold the file
Expand-Archive $zip -DestinationPath $dir -Force
Remove-Item $zip

& (Join-Path $dir 'ghostframe.exe') install

$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if ($userPath -notlike "*$dir*") {
    [Environment]::SetEnvironmentVariable('Path', "$userPath;$dir", 'User')
    Write-Host "Added $dir to your PATH (open a new terminal to use 'ghostframe')."
}
Write-Host ""
Write-Host "Ready. Start a Vulkan game with:  ghostframe run <game.exe>"
Write-Host "Click the ghostframe icon in the notification area to see it and switch it per game."
