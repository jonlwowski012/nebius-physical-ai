"""Encord SaaS seam: auth, domain, public endpoint, and title-or-id resolution.

Everything that talks to the Encord SDK funnels through this module so tests can
monkeypatch ``_default_user_client`` (or inject ``user_client=``) and the
``encord`` package stays a lazy, optional import.
"""

from __future__ import annotations

import base64
import os
import re
from typing import Any

from npa.workbench.encord.schemas import EncordAuthError, EncordToolError

ENCORD_SSH_KEY_ENV = "ENCORD_SSH_KEY"
ENCORD_SSH_KEY_B64_ENV = "ENCORD_SSH_KEY_B64"
ENCORD_SSH_KEY_FILE_ENV = "ENCORD_SSH_KEY_FILE"
ENCORD_DOMAIN_ENV = "ENCORD_DOMAIN"
DEFAULT_ENCORD_DOMAIN = "https://api.encord.com"

AUTH_REMEDY = (
    "Set ENCORD_SSH_KEY (PEM content) or ENCORD_SSH_KEY_B64 (base64 of the PEM) in "
    "the environment or under tokens: in ~/.npa/credentials.yaml, or point "
    "ENCORD_SSH_KEY_FILE at the key file. Generate the key pair in the Encord app "
    "under public keys, and pass the secret to workflow submits with "
    "--secret-env ENCORD_SSH_KEY_B64."
)

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def looks_like_id(value: str) -> bool:
    """Whether ``value`` is UUID/hash-shaped (Encord hashes are UUIDs)."""

    return bool(_UUID_RE.match(value.strip()))


def resolve_domain(environ: dict[str, str] | None = None) -> str:
    env = environ if environ is not None else os.environ
    return env.get(ENCORD_DOMAIN_ENV, "").strip() or DEFAULT_ENCORD_DOMAIN


def resolve_public_endpoint(environ: dict[str, str] | None = None) -> str:
    """Endpoint host used to build the public objectUrls Encord registers."""

    env = environ if environ is not None else os.environ
    endpoint = (
        env.get("AWS_ENDPOINT_URL", "").strip()
        or env.get("NEBIUS_S3_ENDPOINT", "").strip()
    )
    if not endpoint and environ is None:
        from npa.clients.credentials import load_credentials

        endpoint = load_credentials().s3_endpoint.strip()
    if not endpoint:
        raise EncordToolError(
            "No S3 endpoint configured for objectUrl construction. Set "
            "AWS_ENDPOINT_URL / NEBIUS_S3_ENDPOINT or storage.endpoint_url in "
            "~/.npa/credentials.yaml."
        )
    return endpoint.rstrip("/")


def _resolve_auth_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Merge process env with tokens from ~/.npa/credentials.yaml (env wins)."""

    env = dict(environ) if environ is not None else dict(os.environ)
    if environ is None:
        from npa.clients.credentials import load_credentials

        tokens = load_credentials(environ=env).tokens
        for name in (ENCORD_SSH_KEY_ENV, ENCORD_SSH_KEY_B64_ENV, ENCORD_SSH_KEY_FILE_ENV):
            if not env.get(name) and tokens.get(name):
                env[name] = tokens[name]
    return env


def _default_user_client(environ: dict[str, str] | None = None) -> Any:
    """Build an authenticated EncordUserClient from env/NPA credentials."""

    env = _resolve_auth_env(environ)
    ssh_key = env.get(ENCORD_SSH_KEY_ENV, "").strip()
    ssh_key_b64 = env.get(ENCORD_SSH_KEY_B64_ENV, "").strip()
    ssh_key_file = env.get(ENCORD_SSH_KEY_FILE_ENV, "").strip()
    if not (ssh_key or ssh_key_b64 or ssh_key_file):
        raise EncordAuthError(f"No Encord credential found. {AUTH_REMEDY}")
    if not ssh_key and ssh_key_b64:
        try:
            ssh_key = base64.b64decode(ssh_key_b64, validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise EncordAuthError(
                f"ENCORD_SSH_KEY_B64 is not valid base64-encoded UTF-8: {exc}"
            ) from exc

    try:
        from encord.user_client import EncordUserClient
    except ModuleNotFoundError as exc:
        raise EncordToolError(
            "The encord SDK is not installed. Install it with "
            "`pip install 'npa[encord]'` or `pip install encord`."
        ) from exc

    domain = resolve_domain(environ)
    try:
        if ssh_key:
            return EncordUserClient.create_with_ssh_private_key(
                ssh_private_key=ssh_key, domain=domain
            )
        return EncordUserClient.create_with_ssh_private_key(
            ssh_private_key_path=ssh_key_file, domain=domain
        )
    except Exception as exc:  # noqa: BLE001 - SDK raises assorted exception types
        raise EncordAuthError(
            f"Encord authentication failed: {exc}. {AUTH_REMEDY}"
        ) from exc


def resolve_integration(user_client: Any, value: str) -> tuple[str, str]:
    """Resolve an Encord cloud integration by uuid or exact title -> (id, title).

    Integrations are never created here: they hold cloud credentials and must be
    created once in the Encord app (S3-compatible/MinIO pattern for Nebius).
    """

    value = value.strip()
    if not value:
        raise EncordToolError("--integration must not be empty.")
    integrations = list(user_client.get_cloud_integrations())
    if looks_like_id(value):
        for integration in integrations:
            if str(integration.id).lower() == value.lower():
                return str(integration.id), str(integration.title)
        raise EncordToolError(
            f"No Encord cloud integration with id {value!r}. Available titles: "
            f"{sorted(str(i.title) for i in integrations)}"
        )
    matches = [i for i in integrations if str(i.title) == value]
    if len(matches) == 1:
        return str(matches[0].id), str(matches[0].title)
    if not matches:
        raise EncordToolError(
            f"No Encord cloud integration titled {value!r}. Available: "
            f"{sorted(str(i.title) for i in integrations)}. Create an "
            "S3-compatible integration in the Encord app first."
        )
    raise EncordToolError(
        f"Multiple Encord cloud integrations titled {value!r}; pass the "
        "integration id instead."
    )


def resolve_folder(user_client: Any, value: str) -> tuple[Any, bool]:
    """Resolve a storage folder by uuid or title -> (folder, created)."""

    value = value.strip()
    if not value:
        raise EncordToolError("--folder must not be empty.")
    if looks_like_id(value):
        return user_client.get_storage_folder(value), False
    matches = [
        folder
        for folder in user_client.list_storage_folders(search=value, page_size=1000)
        if str(folder.name) == value
    ]
    if len(matches) == 1:
        return matches[0], False
    if len(matches) > 1:
        raise EncordToolError(
            f"Multiple Encord storage folders named {value!r}; pass the folder "
            "uuid instead."
        )
    folder = user_client.create_storage_folder(
        value, description="Created by npa workbench encord push"
    )
    return folder, True


def _dataset_storage_location() -> Any:
    """StorageLocation.CORD_STORAGE, or its name when the SDK is not installed.

    The fallback is only reachable with an injected (fake) user client: any real
    client was constructed by ``_default_user_client``, which already imported
    the SDK.
    """

    try:
        from encord.orm.dataset import StorageLocation
    except ModuleNotFoundError:
        return "CORD_STORAGE"
    return StorageLocation.CORD_STORAGE


def resolve_dataset(
    user_client: Any, value: str, *, create: bool = True
) -> tuple[Any, str, str, bool]:
    """Resolve a dataset by hash or title -> (dataset, hash, title, created)."""

    value = value.strip()
    if not value:
        raise EncordToolError("Dataset reference must not be empty.")
    if looks_like_id(value):
        dataset = user_client.get_dataset(value)
        return dataset, value, str(getattr(dataset, "title", "")), False
    rows = list(user_client.get_datasets(title_eq=value))
    if len(rows) == 1:
        info = rows[0]["dataset"]
        dataset_hash = str(info.dataset_hash)
        return user_client.get_dataset(dataset_hash), dataset_hash, value, False
    if len(rows) > 1:
        raise EncordToolError(
            f"Multiple Encord datasets titled {value!r}; pass the dataset hash "
            "instead."
        )
    if not create:
        raise EncordToolError(f"No Encord dataset titled {value!r}.")

    # Items are registered into our own storage folder and linked explicitly, so
    # the dataset needs no backing folder of its own. CORD_STORAGE is the
    # documented type for link_items-driven datasets; the live e2e smoke is the
    # gate that would surface a mismatch.
    response = user_client.create_dataset(
        value,
        _dataset_storage_location(),
        dataset_description="Created by npa workbench encord push",
        create_backing_folder=False,
    )
    dataset_hash = str(response["dataset_hash"])
    return user_client.get_dataset(dataset_hash), dataset_hash, value, True


def resolve_project(user_client: Any, value: str) -> tuple[Any, str, str]:
    """Resolve a project by hash or title -> (project, hash, title)."""

    value = value.strip()
    if looks_like_id(value):
        project = user_client.get_project(value)
        return project, value, str(getattr(project, "title", ""))
    rows = list(user_client.get_projects(title_eq=value))
    if len(rows) == 1:
        info = rows[0]["project"]
        project_hash = str(info.project_hash)
        return user_client.get_project(project_hash), project_hash, value
    if len(rows) > 1:
        raise EncordToolError(
            f"Multiple Encord projects titled {value!r}; pass the project hash "
            "instead."
        )
    raise EncordToolError(f"No Encord project titled {value!r}.")


def resolve_collection(user_client: Any, value: str) -> tuple[Any, str, str]:
    """Resolve a collection by uuid or name -> (collection, uuid, name)."""

    value = value.strip()
    if looks_like_id(value):
        collection = user_client.get_collection(value)
        return collection, value, str(getattr(collection, "name", ""))
    matches = [
        collection
        for collection in user_client.list_collections()
        if str(collection.name) == value
    ]
    if len(matches) == 1:
        collection = matches[0]
        return collection, str(collection.uuid), value
    if len(matches) > 1:
        raise EncordToolError(
            f"Multiple Encord collections named {value!r}; pass the collection "
            "uuid instead."
        )
    raise EncordToolError(f"No Encord collection named {value!r}.")
