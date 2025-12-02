import sys
import platform
import traceback
import subprocess

print("python:", sys.executable)
print("python version:", sys.version.replace('\n', ' '))
print("platform:", platform.platform())
print("sys.path:")
for p in sys.path:
    print("  ", p)
print("\n--- Attempt import jax ---")
try:
    import jax
    print("jax version:", jax.__version__)
except Exception:
    print("jax import failed:")
    traceback.print_exc()

print("\n--- Attempt import tabpfn ---")
try:
    import tabpfn
    print("tabpfn version:", getattr(tabpfn, '__version__', 'unknown'))
except Exception:
    print("tabpfn import failed:")
    traceback.print_exc()

print("\n--- pip list (top 100 rows) ---")
try:
    out = subprocess.check_output([sys.executable, "-m", "pip", "list", "--format=columns"], stderr=subprocess.STDOUT)
    print(out.decode("utf-8"))
except Exception:
    print("pip list failed:")
    traceback.print_exc()