# Controlling Logging in ML Toolbox

## Understanding Log Messages

The ML Toolbox uses Python's `logging` module to provide information about its operations. The messages you see are **normal and expected** - they indicate the system is working correctly.

### Common Log Messages

#### INFO Level (Informational)
```
INFO:ml_toolbox.computational_kernels.julia_like_kernel:Warming up Julia-like kernel (JIT compilation)...
INFO:ml_toolbox.computational_kernels.julia_like_kernel:Julia-like kernel warmup complete
INFO:ml_toolbox.computational_kernels.unified_computational_kernel:Unified Computational Kernel initialized (mode: auto)
```

**What they mean:**
- ✅ System is initializing computational kernels
- ✅ JIT compilation is warming up (first-time performance optimization)
- ✅ Everything is working correctly

#### WARNING Level (Non-critical issues)
```
Warning: sklearn not available. Install with: pip install scikit-learn
```

**What they mean:**
- ⚠️ Optional dependency is missing
- ⚠️ Some features may be limited
- ⚠️ Core functionality still works

## Controlling Log Verbosity

### Option 1: Suppress INFO Messages (Recommended)

Add this at the start of your script:

```python
import logging

# Suppress INFO messages from ML Toolbox
logging.getLogger('ml_toolbox').setLevel(logging.WARNING)
```

### Option 2: Suppress All INFO Messages

```python
import logging

# Suppress all INFO messages
logging.basicConfig(level=logging.WARNING)
```

### Option 3: Show Only Errors

```python
import logging

# Show only ERROR and CRITICAL messages
logging.basicConfig(level=logging.ERROR)
```

### Option 4: Show Everything (DEBUG mode)

```python
import logging

# Show all messages including DEBUG
logging.basicConfig(level=logging.DEBUG)
```

## Example: Quiet ML Toolbox Usage

```python
import logging

# Suppress INFO messages from ML Toolbox
logging.getLogger('ml_toolbox').setLevel(logging.WARNING)

# Now import and use - no INFO messages
from ml_toolbox import MLToolbox

toolbox = MLToolbox()
# No INFO messages will appear
```

## Example: Suppress Specific Modules

```python
import logging

# Suppress only computational kernels
logging.getLogger('ml_toolbox.computational_kernels').setLevel(logging.WARNING)

# Suppress only agent modules
logging.getLogger('ml_toolbox.ai_agent').setLevel(logging.WARNING)

from ml_toolbox import MLToolbox
toolbox = MLToolbox()
```

## Log Levels Explained

| Level | When to Use | What You See |
|-------|-------------|--------------|
| **DEBUG** | Development | Everything, very verbose |
| **INFO** | Normal use | Important status messages |
| **WARNING** | Production | Only warnings and errors |
| **ERROR** | Troubleshooting | Only errors |
| **CRITICAL** | Minimal | Only critical failures |

## Recommended Settings

### For Development
```python
import logging
logging.basicConfig(level=logging.INFO)  # See what's happening
```

### For Production
```python
import logging
logging.getLogger('ml_toolbox').setLevel(logging.WARNING)  # Quiet
```

### For Troubleshooting
```python
import logging
logging.basicConfig(level=logging.DEBUG)  # See everything
```

## Quick Reference

**Quiet mode (recommended for most users):**
```python
import logging
logging.getLogger('ml_toolbox').setLevel(logging.WARNING)
from ml_toolbox import MLToolbox
```

**Normal mode (default):**
```python
from ml_toolbox import MLToolbox  # Shows INFO messages
```

**Verbose mode (for debugging):**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
from ml_toolbox import MLToolbox
```

## Note

The INFO messages you see are **not errors** - they're just informational messages indicating:
- ✅ System initialization
- ✅ Performance optimizations (JIT warmup)
- ✅ Feature availability

You can safely ignore them or suppress them using the methods above.
