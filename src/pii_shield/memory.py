"""How much room is left before loading another language pipeline.

A pipeline is around a gigabyte resident. Loading a second one inside a container
sized for one does not degrade gracefully: the kernel kills the process, and an agent
that was protected a moment ago is left with no shield at all. Failing closed is the
right instinct, but being OOM-killed is not failing closed — it is failing *gone*.

So the room is measured before the load, the least recently used pipeline is dropped
if that would make space, and if it still would not the request is refused with a
message that says so.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

# Resident size runs above the package size — spaCy's vectors are memory-mapped into
# the process and the analyzer adds its own overhead. Measured at roughly 1.6x for the
# large Russian pipeline, rounded up because being wrong in this direction costs a
# refused request and being wrong in the other costs the whole container.
RESIDENT_TO_DISK_RATIO = 1.8

# Kept free after a load, so the process has room to actually serve a request.
HEADROOM_BYTES = 128 * 1024 * 1024

_LIMIT_FILES = (
    "/sys/fs/cgroup/memory.max",                     # cgroup v2
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",   # cgroup v1
)
_USAGE_FILES = (
    "/sys/fs/cgroup/memory.current",
    "/sys/fs/cgroup/memory/memory.usage_in_bytes",
)

# cgroup v1 reports "no limit" as a number near 2**63 rather than saying so.
_V1_UNLIMITED = 1 << 62


def _read_first(paths) -> int | None:
    for path in paths:
        try:
            raw = Path(path).read_text().strip()
        except (OSError, ValueError):
            continue
        if raw == "max":
            return None
        try:
            value = int(raw)
        except ValueError:
            continue
        return None if value >= _V1_UNLIMITED else value
    return None


def container_limit_bytes() -> int | None:
    """The memory limit this process is held to, or None when there is not one."""
    override = os.environ.get("PII_SHIELD_MEMORY_LIMIT_BYTES")
    if override:
        try:
            return int(override)
        except ValueError:
            return None
    return _read_first(_LIMIT_FILES)


def process_rss_bytes() -> int | None:
    """This process's peak resident size, as a fallback where cgroup files are absent.

    Peak rather than current, which overestimates: the error therefore lands on
    refusing a load that might have fitted, rather than on attempting one that does
    not. ``ru_maxrss`` is kilobytes on Linux and bytes on macOS, hence the switch.
    """
    try:
        import resource
    except ImportError:
        return None
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if not raw:
        return None
    return raw if sys.platform == "darwin" else raw * 1024


def current_usage_bytes() -> int | None:
    return _read_first(_USAGE_FILES) or process_rss_bytes()


def package_size_bytes(name: str) -> int | None:
    """On-disk size of an installed package, used to estimate what loading it costs."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    root = Path(next(iter(spec.submodule_search_locations)))
    try:
        return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    except OSError:
        return None


def estimated_load_bytes(model_name: str) -> int | None:
    size = package_size_bytes(model_name)
    return int(size * RESIDENT_TO_DISK_RATIO) if size else None


def assess(model_name: str) -> tuple[int | None, int | None, str]:
    """(free bytes, bytes the load needs, description). None means "unknown".

    Returned as numbers so a caller can reason about eviction arithmetically. Re-reading
    usage after freeing a pipeline does not work: the fallback metric is *peak* resident
    size, which never goes down, so a loop that waits for the reading to improve evicts
    everything it has and refuses anyway.
    """
    limit = container_limit_bytes()
    if limit is None:
        return None, None, "no memory limit in force"
    usage = current_usage_bytes()
    if usage is None:
        return None, None, "usage unreadable"
    needed = estimated_load_bytes(model_name)
    if needed is None:
        return None, None, "size of the pipeline unknown"
    free = limit - usage
    return free, needed + HEADROOM_BYTES, (
        f"{_mb(free)} free, {_mb(needed)} needed for {model_name} plus "
        f"{_mb(HEADROOM_BYTES)} headroom (limit {_mb(limit)}, in use {_mb(usage)})"
    )


def room_for(model_name: str) -> tuple[bool, str]:
    """Whether *model_name* can be loaded now. Returns (ok, human-readable reason).

    Unknown is treated as fine: a host with no cgroup limit is the ordinary case for a
    workstation, and refusing there would be worse than useless.
    """
    free, needed, reason = assess(model_name)
    if free is None or needed is None:
        return True, reason
    return free >= needed, reason


def _mb(value: int) -> str:
    return f"{value / (1024 * 1024):.0f} MB"
