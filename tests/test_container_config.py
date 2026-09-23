"""The image and the compose file have to agree about what is writable.

The container runs with a read-only root filesystem, and the only writable place is
the tmpfs. Anything that needs to write at runtime must be pointed there deliberately:
tldextract, which Presidio uses to check the TLD of an e-mail, otherwise warns on every
worker start and re-fetches the public suffix list over HTTP — and a cache it finds in
a read-only place raises outright, because it takes a lock file beside the entry before
reading it.
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCKERFILE = (ROOT / "Dockerfile").read_text()
COMPOSE = (ROOT / "docker-compose.yml").read_text()


def tmpfs_mounts() -> list[str]:
    """The paths docker-compose mounts as tmpfs, in either YAML list style."""
    block = re.search(r"^\s*tmpfs:(.*?)(?=^\s*\w[\w-]*:|\Z)", COMPOSE, re.S | re.M)
    assert block, "docker-compose.yml no longer mounts a tmpfs"
    return re.findall(r"[-\[]\s*(/\S+?)\s*[,\]]?$", block.group(1), re.M)


def test_the_root_filesystem_is_read_only():
    assert re.search(r"^\s*read_only:\s*true", COMPOSE, re.M), (
        "the comment in the Dockerfile about a read-only root no longer describes this"
    )


def test_the_suffix_list_cache_is_writable():
    cache = re.search(r"TLDEXTRACT_CACHE=(\S+)", DOCKERFILE)
    assert cache, "TLDEXTRACT_CACHE is gone: Presidio will warn and refetch per worker"
    path = cache.group(1)
    assert any(path == m or path.startswith(m.rstrip("/") + "/") for m in tmpfs_mounts()), (
        f"{path} is not under a tmpfs mount {tmpfs_mounts()}, so it is read-only at runtime"
    )
