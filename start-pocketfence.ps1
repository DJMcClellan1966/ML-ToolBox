# PocketFence Kernel Startup Script
# Starts the PocketFence Kernel service for use with AdvancedDataPreprocessor

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PocketFence Kernel Startup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if .NET is installed
Write-Host "[1/3] Checking .NET installation..." -ForegroundColor Yellow
try {
    $dotnetVersion = dotnet --version
    Write-Host "  .NET Version: $dotnetVersion" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: .NET is not installed!" -ForegroundColor Red
    Write-Host "  Install from: https://dotnet.microsoft.com/download" -ForegroundColor Yellow
    exit 1
}

# Check if PocketFenceKernel directory exists
Write-Host ""
Write-Host "[2/3] Checking PocketFenceKernel directory..." -ForegroundColor Yellow
if (Test-Path "PocketFenceKernel") {
    Write-Host "  Directory found: PocketFenceKernel" -ForegroundColor Green
} else {
    Write-Host "  ERROR: PocketFenceKernel directory not found!" -ForegroundColor Red
    Write-Host "  Make sure you're in the project root directory" -ForegroundColor Yellow
    exit 1
}

# Check if port 5000 is available
Write-Host ""
Write-Host "[3/3] Checking if port 5000 is available..." -ForegroundColor Yellow
$portInUse = Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue
if ($portInUse) {
    Write-Host "  WARNING: Port 5000 is already in use!" -ForegroundColor Yellow
    Write-Host "  Process ID: $($portInUse.OwningProcess)" -ForegroundColor Yellow
    Write-Host "  You may need to stop the existing process first" -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y") {
        exit 0
    }
} else {
    Write-Host "  Port 5000 is available" -ForegroundColor Green
}

# Start the kernel
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting PocketFence Kernel..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "The kernel will start on: http://localhost:5000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the service" -ForegroundColor Yellow
Write-Host ""

# Change to PocketFenceKernel directory and start
Set-Location PocketFenceKernel
dotnet run -- --kernel
