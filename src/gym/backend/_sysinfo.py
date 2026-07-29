"""Minimal sysinfo — avoid psutil dependency. Detect total RAM on the
backends we care about (Darwin / Linux). Returns bytes; rounded to GB at
the call site.
"""

from __future__ import annotations

import os
import platform


def total_ram_bytes() -> int:
    system = platform.system()
    if system == "Darwin":
        return _darwin_memsize()
    if system == "Linux":
        return _linux_memtotal()
    return 8 * 1024**3   # conservative fallback


def _darwin_memsize() -> int:
    try:
        import subprocess
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        return int(out.strip())
    except Exception:
        return 8 * 1024**3


def _linux_memtotal() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb * 1024
    except Exception:
        pass
    return 8 * 1024**3


def apple_chip_name() -> str:
    """Return Apple Silicon chip brand string ('Apple M1 Max', 'Apple M4 Max'...).

    platform.processor() only returns 'arm' on Darwin — useless. The sysctl
    brand string is what carries the variant.
    """
    if platform.system() != "Darwin":
        return ""
    try:
        import subprocess
        out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True)
        return out.strip()
    except Exception:
        return ""


def cpu_brand() -> str:
    """Return the CPU brand string on any OS (or '' if unknown)."""
    if platform.system() == "Darwin":
        return apple_chip_name()
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            return ""
    return platform.processor() or ""


def cpu_core_count() -> int:
    """Logical CPU count (usable for parallel compute)."""
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1


