"""Launch authorization bound to one candidate tree, one GPU UUID and an absolute expiry (A6000 smoke stage).

The repository's `launch_hold` stays active. A launcher may proceed only with a valid, unexpired, unrevoked
authorization that names *this* candidate (path and content digest), the exact GPU index/UUID, the allowed scope
and the GPU/job ceilings. Expiry is an absolute UTC time checked before every launch, so it refuses even if the
controller that issued the exception dies before revoking it. Authorization files are never edited: revocation is
a separate, exclusively created `<file>.revoked`.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

from .safety import UUID_RE

DIGEST_EXCLUDE = {".cache", "__pycache__", ".pytest_cache", ".venv"}
REQUIRED = ("id", "scope", "candidate_root", "candidate_digest", "gpu_index", "gpu_uuid", "max_gpus",
            "max_concurrent_jobs", "max_cases", "issued_utc", "expires_utc", "nonrenewable")


def candidate_digest(root: Path) -> str:
    """SHA-256 over (relative path, file SHA-256) of the candidate tree, excluding caches and result data."""
    root = Path(root)
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if not p.is_file() or set(rel.parts) & DIGEST_EXCLUDE or rel.as_posix().startswith("studies/a6000/results/raw/"):
            continue
        h.update(rel.as_posix().encode() + b"\0" + hashlib.sha256(p.read_bytes()).hexdigest().encode() + b"\n")
    return h.hexdigest()


def parse_utc(s: str) -> dt.datetime:
    t = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise ValueError(f"timestamp without timezone: {s!r}")
    return t.astimezone(dt.timezone.utc)


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def create_exclusive(path: Path, obj: dict) -> None:
    """Write a new JSON record; refuse if anything already exists at `path` (never overwrite a record)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")


def revoke(auth_path: Path, reason: str) -> bool:
    """Exclusively create `<auth>.revoked`. Returns False if it already existed (already revoked)."""
    try:
        create_exclusive(Path(str(auth_path) + ".revoked"), {"revoked_utc": now_utc().isoformat(), "reason": reason})
        return True
    except FileExistsError:
        return False


def authorization_problems(auth_path: Path | None, *, scope: str, candidate_root: Path, gpus: list[int],
                           uuids: dict, n_cases_done: int = 0, now: dt.datetime | None = None,
                           expected_sha256: str | None = None) -> list[str]:
    """Reasons the authorization does NOT permit this launch (empty = permitted)."""
    if auth_path is None:
        return ["no launch authorization given"]
    auth_path = Path(auth_path)
    try:
        raw = auth_path.read_bytes()
        auth = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        return [f"authorization unreadable ({exc!r})"]
    problems = []
    if expected_sha256 and hashlib.sha256(raw).hexdigest() != expected_sha256:
        problems.append("authorization file changed since it was issued")
    missing = [k for k in REQUIRED if k not in auth]
    if missing:
        return problems + [f"authorization malformed: missing {missing}"]
    if Path(str(auth_path) + ".revoked").exists():
        problems.append("authorization revoked")
    now = now or now_utc()
    try:
        issued, expires = parse_utc(auth["issued_utc"]), parse_utc(auth["expires_utc"])
    except Exception as exc:  # noqa: BLE001
        return problems + [f"authorization times malformed ({exc!r})"]
    if not issued <= now < expires:
        problems.append(f"authorization not valid now ({now.isoformat()} outside [{issued.isoformat()}, "
                        f"{expires.isoformat()}))")
    if auth.get("nonrenewable") is not True:
        problems.append("authorization must be nonrenewable")
    if auth["scope"] != scope:
        problems.append(f"authorization scope {auth['scope']!r} does not cover {scope!r}")
    if Path(auth["candidate_root"]).resolve() != Path(candidate_root).resolve():
        problems.append(f"authorization is for candidate {auth['candidate_root']}, not {candidate_root}")
    elif auth["candidate_digest"] != candidate_digest(Path(candidate_root)):
        problems.append("candidate tree changed since authorization (digest mismatch)")
    if not isinstance(auth["max_gpus"], int) or len(gpus) > auth["max_gpus"] or len(gpus) != 1:
        problems.append(f"GPU ceiling: requested {gpus}, authorization allows {auth['max_gpus']} (exactly one)")
    if list(gpus) != [auth["gpu_index"]]:
        problems.append(f"GPU {gpus} is not the authorized index {auth['gpu_index']}")
    if not UUID_RE.match(str(auth["gpu_uuid"])) or uuids.get(auth["gpu_index"]) != auth["gpu_uuid"]:
        problems.append(f"GPU UUID binding mismatch (authorized {auth['gpu_uuid']!r}, pinned {uuids})")
    if n_cases_done >= auth["max_cases"]:
        problems.append(f"case ceiling reached ({n_cases_done} of {auth['max_cases']})")
    return problems
