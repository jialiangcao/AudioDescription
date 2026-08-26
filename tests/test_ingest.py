"""URL ingest: what we ask yt-dlp for, and how its failures surface."""

import sys
from types import SimpleNamespace

import pytest

import ingest
from ingest import (
    IngestError,
    download_youtube,
    is_supported_url,
    probe_youtube,
    video_id,
)


class _FakeYoutubeDL:
    """Records the options it was built with; writes the file it claims to fetch."""

    instances: list["_FakeYoutubeDL"] = []

    def __init__(self, params):
        self.params = params
        self.raised = None
        self.written_suffix = ".mp4"
        _FakeYoutubeDL.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def extract_info(self, url, download=False):
        if download:
            template = self.params["outtmpl"]
            path = template.replace(".%(ext)s", self.written_suffix)
            with open(path, "wb") as handle:
                handle.write(b"x" * 2048)
        return {"id": "abcdefghijk", "title": "A Clip", "duration": 128.0}


class _FakeDownloadError(Exception):
    pass


@pytest.fixture
def fake_ytdlp(monkeypatch):
    """Stand in for the yt_dlp package, which ingest imports lazily."""
    _FakeYoutubeDL.instances = []
    module = SimpleNamespace(YoutubeDL=_FakeYoutubeDL)
    utils = SimpleNamespace(DownloadError=_FakeDownloadError)
    monkeypatch.setitem(sys.modules, "yt_dlp", module)
    monkeypatch.setitem(sys.modules, "yt_dlp.utils", utils)
    return module


# --------------------------------------------------------------------------- #
# URL parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=_SQr8I3lcW8",
        "https://youtu.be/_SQr8I3lcW8",
        "https://m.youtube.com/watch?v=_SQr8I3lcW8",
    ],
)
def test_supported_urls(url):
    assert is_supported_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://vimeo.com/12345",
        # A lookalike host: the check must be on the hostname, not a substring.
        "https://youtube.com.evil.test/watch?v=x",
        "file:///etc/passwd",
        "not a url at all",
    ],
)
def test_unsupported_urls(url):
    assert not is_supported_url(url)


def test_video_id_from_every_accepted_form():
    assert video_id("_SQr8I3lcW8") == "_SQr8I3lcW8"
    assert video_id("https://www.youtube.com/watch?v=_SQr8I3lcW8") == "_SQr8I3lcW8"
    assert video_id("https://youtu.be/_SQr8I3lcW8?t=30") == "_SQr8I3lcW8"
    assert video_id("https://www.youtube.com/watch?list=PL123") is None


# --------------------------------------------------------------------------- #
# probe / download
# --------------------------------------------------------------------------- #


def test_probe_returns_metadata_without_downloading(fake_ytdlp):
    meta = probe_youtube("https://youtu.be/_SQr8I3lcW8")
    assert meta == {
        "id": "abcdefghijk",
        "title": "A Clip",
        "duration_sec": 128.0,
        "extractor": None,
    }


def test_probe_rejects_an_unsupported_host_before_touching_the_network(fake_ytdlp):
    with pytest.raises(IngestError, match="unsupported URL"):
        probe_youtube("https://vimeo.com/12345")
    assert _FakeYoutubeDL.instances == []


def test_download_writes_to_exactly_the_requested_path(fake_ytdlp, tmp_path):
    dest = tmp_path / "clips" / "abcdefghijk.mp4"
    meta = download_youtube("https://youtu.be/_SQr8I3lcW8", dest)

    assert dest.exists()
    assert meta["title"] == "A Clip"
    opts = _FakeYoutubeDL.instances[-1].params
    assert opts["merge_output_format"] == "mp4"
    assert "height<=720" in opts["format"]
    assert opts["noplaylist"] is True


def test_download_caps_resolution_at_max_height(fake_ytdlp, tmp_path):
    download_youtube("https://youtu.be/x", tmp_path / "a.mp4", max_height=360)
    assert "height<=360" in _FakeYoutubeDL.instances[-1].params["format"]


def test_download_moves_a_file_yt_dlp_could_not_remux(fake_ytdlp, tmp_path):
    """merge_output_format is a request, not a guarantee."""
    dest = tmp_path / "clip.mp4"

    original_init = _FakeYoutubeDL.__init__

    def _init(self, params):
        original_init(self, params)
        self.written_suffix = ".webm"

    _FakeYoutubeDL.__init__ = _init
    try:
        download_youtube("https://youtu.be/x", dest)
    finally:
        _FakeYoutubeDL.__init__ = original_init

    assert dest.exists()
    assert not (tmp_path / "clip.webm").exists()


def test_download_error_becomes_ingest_error(fake_ytdlp, tmp_path, monkeypatch):
    def _boom(self, url, download=False):
        raise _FakeDownloadError("Video unavailable")

    monkeypatch.setattr(_FakeYoutubeDL, "extract_info", _boom)
    with pytest.raises(IngestError, match="Video unavailable"):
        download_youtube("https://youtu.be/x", tmp_path / "a.mp4")


def test_cookie_env_is_passed_through(fake_ytdlp, monkeypatch):
    monkeypatch.setenv("ADESC_YTDLP_COOKIES_FROM_BROWSER", "chrome")
    probe_youtube("https://youtu.be/x")
    assert _FakeYoutubeDL.instances[-1].params["cookiesfrombrowser"] == ("chrome",)


def test_yt_dlp_is_not_imported_at_module_import():
    """The slim image has no yt-dlp; tasks.py imports ingest and must still load."""
    assert "yt_dlp" not in dir(ingest)
