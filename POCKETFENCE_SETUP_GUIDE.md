# PocketFence Kernel Setup Guide

## Why PocketFence Kernel Isn't Available

The **PocketFence Kernel** is a **separate .NET service** that needs to be **started manually**. It's not automatically running, which is why the tests show:

```
[Warning] PocketFence service not available at http://localhost:5000
```

---

## What is PocketFence Kernel?

PocketFence Kernel is a **content filtering and safety checking service** that provides:
- ✅ **Advanced threat detection** (spam, malware, phishing)
- ✅ **URL safety validation**
- ✅ **Content safety scoring**
- ✅ **Real-time filtering**
- ✅ **REST API** for integration

---

## How to Start PocketFence Kernel

### Option 1: Run from Source (Recommended)

```bash
# Navigate to PocketFenceKernel directory
cd PocketFenceKernel

# Start the kernel API server
dotnet run -- --kernel
```

**Expected Output:**
```
info: Microsoft.Hosting.Lifetime[14]
      Now listening on: http://localhost:5000
info: Microsoft.Hosting.Lifetime[0]
      Application started. Press Ctrl+C to shut down.
```

### Option 2: Build and Run

```bash
# Build the project
cd PocketFenceKernel
dotnet build

# Run in kernel mode
dotnet run -- --kernel
```

### Option 3: Run as Windows Service

```bash
# Create Windows service
sc create PocketFenceKernel binpath="C:\path\to\PocketFence-AI.exe --service"
sc start PocketFenceKernel
```

### Option 4: Docker (if configured)

```bash
# Build Docker image
docker build -t pocketfence-kernel .

# Run container
docker run -d -p 5000:5000 pocketfence-kernel
```

---

## Verify PocketFence is Running

### Check Health Endpoint

```bash
# Using curl
curl http://localhost:5000/api/kernel/health

# Using PowerShell
Invoke-WebRequest -Uri http://localhost:5000/api/kernel/health

# Using Python
import requests
response = requests.get("http://localhost:5000/api/kernel/health")
print(response.status_code)  # Should be 200
```

**Expected Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": "00:05:23"
}
```

### Test Content Filtering

```bash
# Test URL filtering
curl -X POST http://localhost:5000/api/filter/url \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com"}'

# Test content filtering
curl -X POST http://localhost:5000/api/filter/content \
  -H "Content-Type: application/json" \
  -d '{"content":"This is spam content"}'
```

---

## Configuration

### Default Settings

The service runs on **port 5000** by default. You can change this in `appsettings.json`:

```json
{
  "Kestrel": {
    "Endpoints": {
      "Http": {
        "Url": "http://localhost:5000"
      }
    }
  }
}
```

### Update AdvancedDataPreprocessor URL

If you change the port, update the preprocessor:

```python
from data_preprocessor import AdvancedDataPreprocessor

# Use custom port
preprocessor = AdvancedDataPreprocessor(
    pocketfence_url="http://localhost:8080"  # Custom port
)
```

---

## Troubleshooting

### Issue 1: Port Already in Use

**Error:** `Address already in use` or `Only one usage of each socket address`

**Solution:**
```bash
# Find process using port 5000
netstat -ano | findstr :5000

# Kill the process (Windows)
taskkill /PID <process_id> /F

# Or use a different port
# Edit appsettings.json to use port 5001
```

### Issue 2: .NET Not Installed

**Error:** `'dotnet' is not recognized`

**Solution:**
1. Install [.NET 8.0 SDK](https://dotnet.microsoft.com/download)
2. Verify installation:
   ```bash
   dotnet --version
   ```

### Issue 3: Build Errors

**Error:** Build fails with missing dependencies

**Solution:**
```bash
# Restore packages
dotnet restore

# Clean and rebuild
dotnet clean
dotnet build
```

### Issue 4: Service Won't Start

**Error:** Service fails to start

**Solution:**
```bash
# Check logs
dotnet run -- --kernel 2>&1 | tee kernel.log

# Check for port conflicts
netstat -ano | findstr :5000

# Try different port
# Edit appsettings.json
```

---

## Running PocketFence in Background

### Windows (PowerShell)

```powershell
# Start in background
Start-Process -NoNewWindow dotnet -ArgumentList "run -- --kernel" -WorkingDirectory "PocketFenceKernel"

# Or use Start-Job
$job = Start-Job -ScriptBlock {
    cd PocketFenceKernel
    dotnet run -- --kernel
}
```

### Linux/macOS

```bash
# Start in background
cd PocketFenceKernel
nohup dotnet run -- --kernel > kernel.log 2>&1 &

# Check if running
ps aux | grep dotnet
```

---

## Integration with AdvancedDataPreprocessor

Once PocketFence is running, the AdvancedDataPreprocessor will automatically use it:

```python
from data_preprocessor import AdvancedDataPreprocessor

# PocketFence will be used automatically if available
preprocessor = AdvancedDataPreprocessor(
    pocketfence_url="http://localhost:5000"  # Default
)

# Preprocess with safety filtering
results = preprocessor.preprocess(raw_data, verbose=True)

# Check if unsafe content was filtered
print(f"Unsafe items filtered: {len(results['unsafe_data'])}")
```

---

## API Endpoints

When PocketFence is running, these endpoints are available:

### Health Check
```
GET http://localhost:5000/api/kernel/health
```

### Filter URL
```
POST http://localhost:5000/api/filter/url
Content-Type: application/json

{
  "url": "https://example.com"
}
```

### Filter Content
```
POST http://localhost:5000/api/filter/content
Content-Type: application/json

{
  "content": "Text to check"
}
```

### Swagger Documentation
```
http://localhost:5000/swagger
```

---

## Quick Start Script

Create a startup script to make it easier:

### Windows (`start-pocketfence.bat`)
```batch
@echo off
cd PocketFenceKernel
echo Starting PocketFence Kernel...
dotnet run -- --kernel
pause
```

### Linux/macOS (`start-pocketfence.sh`)
```bash
#!/bin/bash
cd PocketFenceKernel
echo "Starting PocketFence Kernel..."
dotnet run -- --kernel
```

Make executable:
```bash
chmod +x start-pocketfence.sh
```

---

## Summary

**Why it's not available:**
- PocketFence Kernel is a separate service that needs to be started
- It's not running by default
- The service must be started before use

**How to start it:**
```bash
cd PocketFenceKernel
dotnet run -- --kernel
```

**Verify it's running:**
```bash
curl http://localhost:5000/api/kernel/health
```

**Once running:**
- AdvancedDataPreprocessor will automatically use it
- Safety filtering will work
- Tests will show real filtering results

---

**Start the PocketFence Kernel service to enable advanced safety filtering!**
