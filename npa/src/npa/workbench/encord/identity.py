"""Exact identity for Encord storage items (adopted from PR #363).

Display fields are deliberately absent from this module: a filename or title
is never identity. Identity comes from the namespaced ``npa.source_uri``
clientMetadata registered with the item, or from the item's complete
normalized object URL. Resolution is exact, and conflicting signals fail
closed rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping
from urllib.parse import quote, urlsplit, urlunsplit

from npa.workbench.encord.schemas import EncordToolError


def canonical_s3_uri(bucket: str, key: str) -> str:
    """The canonical s3:// identity of one object, safe against key aliasing."""

    bucket = bucket.strip()
    if not bucket or any(char in bucket for char in "/\\\x00:@?#"):
        raise EncordToolError("S3 bucket must be a nonempty bucket name")
    if not key or key.endswith("/") or "\\" in key or "\x00" in key:
        raise EncordToolError("S3 object key must identify one unambiguous object")
    if key.startswith("/") or any(part in {"", ".", ".."} for part in key.split("/")):
        raise EncordToolError("S3 object key contains an ambiguous path form")
    # The SDK returns literal object keys. Encode the percent sign instead of
    # interpreting percent triplets, so the key ``a%2Fb`` never aliases ``a/b``.
    encoded = quote(key, safe="/~:@!$&'()*+,;=-._")
    return f"s3://{bucket}/{encoded}"


def normalize_object_url(url: str) -> str:
    """One canonical form per object URL, so comparison means identity."""

    parsed = urlsplit(url.strip())
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise EncordToolError("object URL must be an absolute HTTP(S) URL")
    host = parsed.hostname.lower()
    try:
        port = parsed.port
    except ValueError as exc:
        raise EncordToolError("object URL contains an invalid port") from exc
    if port:
        host = f"{host}:{port}"
    normalized_path = _normalize_url_path(parsed.path)
    segments = normalized_path.split("/")[1:]
    if not normalized_path.startswith("/") or any(
        segment in {"", ".", ".."} for segment in segments
    ):
        raise EncordToolError("object URL contains an ambiguous path")
    return urlunsplit((parsed.scheme.lower(), host, normalized_path, "", ""))


def _normalize_url_path(path: str) -> str:
    """Normalize only unreserved escapes while preserving reserved identity."""

    unreserved = frozenset(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
    )
    hexadecimal = frozenset("0123456789abcdefABCDEF")
    normalized: list[str] = []
    index = 0
    while index < len(path):
        character = path[index]
        if character == "%":
            if index + 2 >= len(path) or not {
                path[index + 1],
                path[index + 2],
            } <= hexadecimal:
                raise EncordToolError("object URL contains an invalid percent escape")
            value = int(path[index + 1 : index + 3], 16)
            decoded = chr(value)
            if decoded in {"\\", "\x00"}:
                raise EncordToolError("object URL contains an ambiguous path")
            normalized.append(decoded if decoded in unreserved else f"%{value:02X}")
            index += 3
            continue
        if character in {"\\", "\x00"}:
            raise EncordToolError("object URL contains an ambiguous path")
        normalized.append(quote(character, safe="/~:@!$&'()*+,;=-._"))
        index += 1
    return "".join(normalized)


def identity_metadata(source_uri: str) -> dict[str, Any]:
    """The namespaced clientMetadata payload registered with every item."""

    return {"npa": {"source_uri": source_uri}}


def metadata_identity(item: Any) -> str:
    """The item's npa.source_uri metadata identity, or empty."""

    raw = getattr(item, "client_metadata", None) or {}
    if not isinstance(raw, Mapping):
        return ""
    namespace = raw.get("npa") or {}
    if not isinstance(namespace, Mapping):
        return ""
    return str(namespace.get("source_uri") or "").strip()


def object_url_identity(item: Any) -> str:
    for attr in ("object_url", "objectUrl", "url", "file_url"):
        raw = getattr(item, attr, "")
        if raw:
            try:
                return normalize_object_url(str(raw))
            except EncordToolError:
                return ""
    return ""


@dataclass(frozen=True)
class IdentityCandidate:
    item_uuid: str
    source_uri: str = ""
    object_url: str = ""

    @classmethod
    def from_item(cls, item: Any) -> "IdentityCandidate":
        item_uuid = str(
            getattr(item, "uuid", "") or getattr(item, "item_uuid", "") or ""
        ).strip()
        return cls(
            item_uuid=item_uuid,
            source_uri=metadata_identity(item),
            object_url=object_url_identity(item),
        )


@dataclass(frozen=True)
class IdentityResolution:
    item_uuid: str = ""
    signal: str = ""
    error_code: str = ""
    error: str = ""

    @property
    def resolved(self) -> bool:
        return bool(self.item_uuid) and not self.error_code


def resolve_exact_identity(
    *,
    source_uri: str,
    submitted_object_url: str,
    candidates: Iterable[Any],
) -> IdentityResolution:
    """Resolve one source object to at most one Encord item, or fail closed."""

    expected_url = (
        normalize_object_url(submitted_object_url) if submitted_object_url else ""
    )
    views = [IdentityCandidate.from_item(candidate) for candidate in candidates]
    views = [view for view in views if view.item_uuid]

    matched: list[tuple[IdentityCandidate, str]] = []
    for view in views:
        if source_uri and view.source_uri == source_uri:
            matched.append((view, "metadata"))
        elif expected_url and view.object_url == expected_url:
            matched.append((view, "object_url"))

    uuids = {view.item_uuid for view, _ in matched}
    conflicts: list[str] = []
    for view in views:
        if view.item_uuid not in uuids:
            continue
        if view.source_uri and view.source_uri != source_uri:
            conflicts.append(f"{view.item_uuid}:source_uri")
        if expected_url and view.object_url and view.object_url != expected_url:
            conflicts.append(f"{view.item_uuid}:object_url")
    if conflicts or len(uuids) > 1:
        detail = ", ".join(sorted(conflicts)) or "multiple exact UUID candidates"
        return IdentityResolution(
            error_code="identity_conflict",
            error=f"exact identity signals conflict: {detail}",
        )
    if not matched:
        return IdentityResolution(
            error_code="identity_unresolved",
            error="no exact metadata or object URL identity matched",
        )
    order = {"metadata": 0, "object_url": 1}
    view, signal = sorted(matched, key=lambda entry: order[entry[1]])[0]
    return IdentityResolution(item_uuid=view.item_uuid, signal=signal)
