"""Locate test fixtures, fetching the large ones on demand.

Small fixtures are committed under ``test/data/`` and found there. The
rest are listed in ``fixtures.toml``, downloaded once, and cached under
``~/.cache/clodius/fixtures``. Nothing is fetched unless a test asks for it, and
a test that cannot get its fixture is skipped rather than failed -- so the suite
stays green offline and on a fresh checkout.

Set ``CLODIUS_FIXTURE_CACHE`` to relocate the cache, ``CLODIUS_FIXTURE_BASE_URL``
to fetch from a mirror, or ``CLODIUS_NO_DOWNLOAD=1`` to skip anything not
already cached.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

TEST_DIR = Path(__file__).parent
MANIFEST = TEST_DIR / "fixtures.toml"
COMMITTED = TEST_DIR / "data"

# Read as it is written, then converted to a MiB-ish chunk for the copy loop.
_CHUNK = 1 << 20


class FixtureError(Exception):
    """A fixture could not be located, fetched or verified."""


def cache_dir() -> Path:
    """Where fetched fixtures live between runs."""
    override = os.environ.get("CLODIUS_FIXTURE_CACHE")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or "~/.cache"
    return Path(base).expanduser() / "clodius" / "fixtures"


def load_manifest() -> dict:
    with open(MANIFEST, "rb") as f:
        return tomllib.load(f)


def _base_url(manifest: dict) -> str:
    return os.environ.get(
        "CLODIUS_FIXTURE_BASE_URL", manifest["base_url"]
    ).rstrip("/")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(path: Path, expected: str) -> None:
    """Check a file against its recorded digest.

    An empty digest means "not pinned yet", which is how a fixture looks while
    it is being built. Verification is skipped rather than failed, so the
    manifest can be filled in after the file exists.
    """
    if not expected:
        return
    actual = digest(path)
    if actual != expected:
        raise FixtureError(
            f"{path} has digest {actual}, expected {expected}. The cached copy "
            f"is stale or the upload changed; delete it and retry."
        )


_USER_AGENT = "clodius-test-fixtures"


def _download(url: str, dest: Path) -> None:
    """Fetch to a temporary sibling, then rename.

    Renaming last means an interrupted download never leaves a short file in
    the cache for the next run to trust.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    # Any agent but the default: Cloudflare's bot protection sits in front of
    # `r2.dev` and answers `Python-urllib/*` with 403, whatever the bucket's
    # permissions say.
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request) as response, open(tmp, "wb") as out:
            shutil.copyfileobj(response, out, _CHUNK)
    except (urllib.error.URLError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise FixtureError(f"could not fetch {url}: {exc}") from exc
    tmp.rename(dest)


def path(name: str) -> Path:
    """The local path to a fixture, fetching it if it is not already here.

    Parameters
    ----------
    name :
        A path under ``test/data`` for a committed fixture, or a key in
        ``fixtures.toml`` for a fetched one.

    Returns
    -------
    Path

    Raises
    ------
    FixtureError
        If the fixture is neither committed, cached, nor fetchable. Tests
        should use :func:`require` instead, which turns this into a skip.
    """
    committed = COMMITTED / name
    if committed.exists():
        return committed

    manifest = load_manifest()
    entry = manifest["files"].get(name)
    if entry is None:
        raise FixtureError(
            f"{name!r} is neither in {COMMITTED} nor listed in {MANIFEST.name}"
        )

    cached = cache_dir() / name
    # Companions -- an index beside its data file -- are fetched with it, since
    # a reader needs them together and a test never asks for them by name.
    for companion in entry.get("companions", []):
        path(companion)

    if cached.exists():
        _verify(cached, entry.get("sha256", ""))
        return cached

    if os.environ.get("CLODIUS_NO_DOWNLOAD"):
        raise FixtureError(
            f"{name!r} is not cached and CLODIUS_NO_DOWNLOAD is set"
        )

    url = f"{_base_url(manifest)}/{name}"
    _download(url, cached)
    _verify(cached, entry.get("sha256", ""))
    return cached


def require(name: str) -> Path:
    """:func:`path`, but skip the test instead of failing it."""
    import pytest

    try:
        return path(name)
    except FixtureError as exc:
        pytest.skip(str(exc))


def _stanza(source: str, key: str) -> str:
    """The manifest entry for a file, ready to paste."""
    p = Path(source)
    return (
        f'[files."{key}"]\n'
        f'sha256 = "{digest(p)}"\n'
        f"size = {p.stat().st_size}\n"
        f'description = ""\n'
    )


def _main(argv: list[str]) -> int:
    usage = (
        "usage:\n"
        "  python test/fixtures.py stanza <file> <key>   manifest entry\n"
        "  python test/fixtures.py fetch [<key> ...]     pre-warm the cache\n"
        "  python test/fixtures.py list                  what is listed\n"
    )
    if not argv or argv[0] in ("-h", "--help"):
        print(usage)
        return 0

    command, *rest = argv
    if command == "stanza":
        if len(rest) != 2:
            print(usage)
            return 2
        print(_stanza(*rest))
    elif command == "fetch":
        keys = rest or list(load_manifest()["files"])
        for key in keys:
            print(f"{key} -> {path(key)}")
    elif command == "list":
        for key, entry in load_manifest()["files"].items():
            where = "cached" if (cache_dir() / key).exists() else "remote"
            print(f"{where:>7}  {key}  {entry.get('size', 0):>12,} B")
    else:
        print(usage)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
