import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

PACKAGES = [
    "numpy", "scipy", "scikit-learn", "pandas", "pyarrow", "openml", "statsmodels",
    "baycomp", "scikit-posthocs", "threadpoolctl", "matplotlib", "pyyaml", "pytest",
    "torch", "faiss-cpu",
]


def cpu_name():
    if platform.system() == "Windows":
        cmd = ["powershell", "-NoProfile", "-Command",
               "(Get-CimInstance Win32_Processor | Select-Object -First 1 Name,NumberOfCores,"
               "NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Json)"]
        try:
            return json.loads(subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout)
        except Exception:
            return platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None

    # numpy and scikit-learn are imported so that their BLAS and OpenMP runtimes are loaded and reported.
    import numpy  # noqa: F401
    import sklearn.neighbors  # noqa: F401
    from threadpoolctl import threadpool_info

    record = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(),
        "packages": versions,
        "threadpools": [
            {k: v for k, v in pool.items() if k in ("user_api", "internal_api", "num_threads", "version", "filepath")}
            for pool in threadpool_info()
        ],
    }
    out = json.dumps(record, indent=2, default=str)
    print(out)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
        Path(sys.argv[1]).write_text(out + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
