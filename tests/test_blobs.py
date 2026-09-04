import pytest

from blobs import JobBlobs


class FakeStore:
    """Stand-in for the boto3 S3 client: an in-memory bucket + call counts."""

    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.downloads = []
        self.uploads = []

    def download_file(self, bucket, key, dest):
        self.downloads.append(key)
        if key not in self.objects:
            raise KeyError(key)
        with open(dest, "wb") as out:
            out.write(self.objects[key])

    def upload_file(self, src, bucket, key):
        self.uploads.append(key)
        with open(src, "rb") as f:
            self.objects[key] = f.read()

    def head_object(self, Bucket, Key):  # noqa: N803 - boto3's kwarg casing
        if Key not in self.objects:
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {"ContentLength": len(self.objects[Key])}

    def generate_presigned_url(self, op, Params, ExpiresIn):  # noqa: N803
        return f"https://bucket.example/{Params['Key']}?op={op}&ttl={ExpiresIn}"

    def get_paginator(self, op):
        store = self

        class _Paginator:
            def paginate(self, Bucket, Prefix):  # noqa: N803
                yield {
                    "Contents": [
                        {"Key": key}
                        for key in sorted(store.objects)
                        if key.startswith(Prefix)
                    ]
                }

        return _Paginator()

    def delete_objects(self, Bucket, Delete):  # noqa: N803
        for obj in Delete["Objects"]:
            self.objects.pop(obj["Key"], None)


@pytest.fixture
def bucket_env(monkeypatch):
    monkeypatch.setenv("R2_BUCKET", "test-bucket")


def test_object_key_namespaces_by_job(tmp_path):
    blobs = JobBlobs("job-1", root=tmp_path)
    assert blobs.object_key("frames/shot_0000_00.jpg") == (
        "jobs/job-1/frames/shot_0000_00.jpg"
    )


def test_path_creates_parent_dirs(tmp_path):
    blobs = JobBlobs("job-1", root=tmp_path)
    path = blobs.path("narration/shot_0000.wav")

    assert path == tmp_path / "narration" / "shot_0000.wav"
    assert path.parent.is_dir()


def test_fetch_returns_scratch_copy_without_touching_the_bucket(tmp_path, bucket_env):
    store = FakeStore()
    blobs = JobBlobs("job-1", root=tmp_path, store=store)
    blobs.path("frames/a.jpg").write_bytes(b"already here")

    assert blobs.fetch("frames/a.jpg").read_bytes() == b"already here"
    assert store.downloads == []


def test_fetch_downloads_once_then_serves_from_scratch(tmp_path, bucket_env):
    store = FakeStore({"jobs/job-1/frames/a.jpg": b"remote bytes"})
    blobs = JobBlobs("job-1", root=tmp_path, store=store)

    assert blobs.fetch("frames/a.jpg").read_bytes() == b"remote bytes"
    assert blobs.fetch("frames/a.jpg").read_bytes() == b"remote bytes"
    # The second call is a cache hit, which is what keeps a multi-stage job from
    # re-downloading every frame it already has.
    assert store.downloads == ["jobs/job-1/frames/a.jpg"]


def test_failed_download_leaves_no_partial_file_to_be_cached(tmp_path, bucket_env):
    """A crashed transfer must not leave a truncated file that later reads trust."""
    store = FakeStore()  # the object does not exist
    blobs = JobBlobs("job-1", root=tmp_path, store=store)

    with pytest.raises(KeyError):
        blobs.fetch("frames/missing.jpg")

    assert not blobs.path("frames/missing.jpg").exists()


def test_fetch_without_a_bucket_raises_for_a_missing_key(tmp_path):
    blobs = JobBlobs("job-1", root=tmp_path)

    with pytest.raises(FileNotFoundError):
        blobs.fetch("frames/never-written.jpg")


def test_put_uploads_under_the_job_namespace(tmp_path, bucket_env):
    store = FakeStore()
    blobs = JobBlobs("job-1", root=tmp_path, store=store)
    blobs.path("described.mp4").write_bytes(b"video bytes")

    assert blobs.put("described.mp4") == "described.mp4"
    assert store.objects["jobs/job-1/described.mp4"] == b"video bytes"


def test_put_is_a_noop_without_a_bucket(tmp_path):
    """Local runs and tests write straight to scratch and never upload."""
    blobs = JobBlobs("job-1", root=tmp_path)
    blobs.path("described.mp4").write_bytes(b"video bytes")

    assert blobs.put("described.mp4") == "described.mp4"


def test_exists_checks_scratch_then_the_bucket(tmp_path, bucket_env):
    store = FakeStore({"jobs/job-1/remote.mp4": b"x"})
    blobs = JobBlobs("job-1", root=tmp_path, store=store)
    blobs.path("local.mp4").write_bytes(b"x")

    assert blobs.exists("local.mp4")
    assert blobs.exists("remote.mp4")
    assert not blobs.exists("nowhere.mp4")


def test_size_reports_none_for_an_absent_object(tmp_path, bucket_env):
    store = FakeStore({"jobs/job-1/there.mp4": b"12345"})
    blobs = JobBlobs("job-1", root=tmp_path, store=store)

    assert blobs.size("there.mp4") == 5
    assert blobs.size("absent.mp4") is None


def test_presigning_requires_a_bucket(tmp_path):
    blobs = JobBlobs("job-1", root=tmp_path)

    with pytest.raises(RuntimeError):
        blobs.presign_get("frames/a.jpg")
    with pytest.raises(RuntimeError):
        blobs.presign_put("source.mp4")


def test_presigned_urls_address_the_namespaced_key(tmp_path, bucket_env):
    blobs = JobBlobs("job-1", root=tmp_path, store=FakeStore())

    assert "jobs/job-1/frames/a.jpg" in blobs.presign_get("frames/a.jpg", ttl_sec=60)
    assert "op=put_object" in blobs.presign_put("source.mp4")


def test_delete_all_clears_the_bucket_prefix_and_scratch(tmp_path, bucket_env):
    """Deleting a job must not leave another job's objects behind."""
    store = FakeStore(
        {
            "jobs/job-1/frames/a.jpg": b"a",
            "jobs/job-1/narration/shot_0000.wav": b"b",
            "jobs/job-2/frames/a.jpg": b"someone else's",
        }
    )
    blobs = JobBlobs("job-1", root=tmp_path, store=store)
    blobs.path("frames/a.jpg").write_bytes(b"a")

    removed = blobs.delete_all()

    assert removed == 2
    assert set(store.objects) == {"jobs/job-2/frames/a.jpg"}
    assert not tmp_path.exists()


def test_delete_all_without_a_bucket_only_clears_scratch(tmp_path):
    blobs = JobBlobs("job-1", root=tmp_path)
    blobs.path("frames/a.jpg").write_bytes(b"a")

    assert blobs.delete_all() == 0
    assert not tmp_path.exists()
