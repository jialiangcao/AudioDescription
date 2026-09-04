"""Job-scoped blob storage: local scratch dir in front of an S3-compatible store.

Every artifact the pipeline produces (sampled frames, narration clips, the AD
track, the described video) is addressed by a **job-relative key** such as
``frames/shot_0000_00.jpg`` or ``narration/ad_track.wav``. Keys are what the
``Timeline`` carries, so a timeline is portable across machines and safe to hand
to a browser — unlike the absolute ``/tmp`` paths it used to hold.

``JobBlobs`` binds a key namespace to one local scratch directory and,
optionally, one remote bucket:

- ``path(key)``  — where the key lives locally (parents created).
- ``fetch(key)`` — a readable local path, downloading from the bucket only if
  the file isn't already in scratch. Stages therefore never re-download what
  they just wrote, and a worker that picks up a later stage pulls exactly the
  inputs it needs.
- ``put(key)``   — upload what's in scratch to the bucket.
- ``presign_get(key)`` — a short-lived URL the browser can fetch directly, so
  media bytes never flow through the API.

``store=None`` gives a purely local instance with no bucket at all. That is what
``pipeline.run_pipeline`` and the test-suite use, so neither needs credentials
or a running MinIO; uploads become no-ops and ``fetch`` is just ``path``.
"""

import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# Presigned URLs are handed to a browser that may sit on the page for a while
# (a long video keeps streaming), but should not outlive a session.
DEFAULT_PRESIGN_TTL_SEC = 3600

# Where a worker keeps its per-job scratch. On Fly this points at a volume.
SCRATCH_ROOT = Path(os.environ.get("ADESC_SCRATCH_ROOT", "/tmp/adesc-jobs"))

_client = None
_client_lock = threading.Lock()


def bucket_name() -> str | None:
    return os.environ.get("R2_BUCKET")


def s3_client():
    """The shared boto3 S3 client, or None when object storage isn't configured.

    Built lazily and cached: boto3 clients are thread-safe for calls but
    expensive to construct, and the API process may never touch one.
    """
    global _client
    if _client is not None:
        return _client
    endpoint = os.environ.get("S3_ENDPOINT")
    if not endpoint or not bucket_name():
        return None
    with _client_lock:
        if _client is None:
            import boto3
            from botocore.config import Config

            _client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=os.environ.get("S3_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("S3_SECRET_ACCESS_KEY"),
                # R2 only implements the v4 signer and has no meaningful region.
                region_name=os.environ.get("S3_REGION", "auto"),
                config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
            )
            logger.info("s3: client ready for bucket %s at %s", bucket_name(), endpoint)
    return _client


class JobBlobs:
    """Key -> bytes for one job, backed by scratch and (optionally) a bucket."""

    def __init__(self, job_id: str, root: Path | None = None, store=None):
        self.job_id = job_id
        self.root = Path(root) if root is not None else SCRATCH_ROOT / job_id
        self.root.mkdir(parents=True, exist_ok=True)
        self._store = store

    @classmethod
    def remote(cls, job_id: str, root: Path | None = None) -> "JobBlobs":
        """A job bound to the configured bucket (falls back to local-only)."""
        return cls(job_id, root=root, store=s3_client())

    # -- keys ----------------------------------------------------------------

    def object_key(self, key: str) -> str:
        """The bucket-absolute key for a job-relative one."""
        return f"jobs/{self.job_id}/{key}"

    def path(self, key: str) -> Path:
        """The local scratch path for ``key``, with its parent created."""
        local = self.root / key
        local.parent.mkdir(parents=True, exist_ok=True)
        return local

    # -- transfer ------------------------------------------------------------

    def fetch(self, key: str) -> Path:
        """A readable local path for ``key``, downloading it only if missing."""
        local = self.path(key)
        if local.exists():
            return local
        if self._store is None:
            raise FileNotFoundError(
                f"{key} is not in scratch and no bucket is configured"
            )
        logger.debug("blobs: fetching %s", key)
        # Download to a sibling temp name first so a crashed transfer can't
        # leave a truncated file that a later fetch would treat as a cache hit.
        tmp = local.with_suffix(local.suffix + ".part")
        self._store.download_file(bucket_name(), self.object_key(key), str(tmp))
        tmp.replace(local)
        return local

    def put(self, key: str) -> str:
        """Upload scratch's copy of ``key`` to the bucket. No-op when local-only."""
        if self._store is None:
            return key
        local = self.path(key)
        logger.debug("blobs: uploading %s (%d bytes)", key, local.stat().st_size)
        self._store.upload_file(str(local), bucket_name(), self.object_key(key))
        return key

    def put_many(self, keys: list[str]) -> list[str]:
        for key in keys:
            self.put(key)
        return keys

    def exists(self, key: str) -> bool:
        if self.path(key).exists():
            return True
        if self._store is None:
            return False
        from botocore.exceptions import ClientError

        try:
            self._store.head_object(Bucket=bucket_name(), Key=self.object_key(key))
        except ClientError:
            return False
        return True

    def size(self, key: str) -> int | None:
        """Remote object size in bytes, or None if it isn't in the bucket."""
        if self._store is None:
            local = self.path(key)
            return local.stat().st_size if local.exists() else None
        from botocore.exceptions import ClientError

        try:
            head = self._store.head_object(
                Bucket=bucket_name(), Key=self.object_key(key)
            )
        except ClientError:
            return None
        return head["ContentLength"]

    def delete_all(self) -> int:
        """Remove every object this job owns, in the bucket and in scratch.

        Deleting a job has to take its media with it — the R2 lifecycle rule on
        ``jobs/*`` expires artifacts eventually, but a user who deletes a video
        means now. Returns the number of objects removed from the bucket.

        The listing is paginated because a long video's frames run to thousands
        of keys, and ``delete_objects`` takes at most 1000 at a time.
        """
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)
        if self._store is None:
            return 0

        prefix = f"jobs/{self.job_id}/"
        removed = 0
        paginator = self._store.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name(), Prefix=prefix):
            batch = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
            if not batch:
                continue
            self._store.delete_objects(
                Bucket=bucket_name(), Delete={"Objects": batch, "Quiet": True}
            )
            removed += len(batch)
        logger.info("blobs: deleted %d object(s) under %s", removed, prefix)
        return removed

    # -- presigning ----------------------------------------------------------

    def presign_get(self, key: str, ttl_sec: int = DEFAULT_PRESIGN_TTL_SEC) -> str:
        """A short-lived URL the browser can GET directly from the bucket."""
        if self._store is None:
            raise RuntimeError("presign_get requires object storage to be configured")
        return self._store.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name(), "Key": self.object_key(key)},
            ExpiresIn=ttl_sec,
        )

    def presign_put(
        self,
        key: str,
        ttl_sec: int = DEFAULT_PRESIGN_TTL_SEC,
        content_type: str | None = None,
    ) -> str:
        """A short-lived URL the browser can PUT the source video to."""
        if self._store is None:
            raise RuntimeError("presign_put requires object storage to be configured")
        params = {"Bucket": bucket_name(), "Key": self.object_key(key)}
        if content_type:
            params["ContentType"] = content_type
        return self._store.generate_presigned_url(
            "put_object", Params=params, ExpiresIn=ttl_sec
        )
