import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "data_downloader",
    Path(__file__).resolve().parents[1] / "scripts" / "download_data.py",
)
# The offline adapter tests do not require importing the optional native engine.
import sys
from types import ModuleType

info = ModuleType("alfworld.info")
info.ALFRED_PDDL_PATH = "fixture"
info.ALFRED_TWL2_PATH = "fixture"
with patch.dict(sys.modules, {"alfworld.info": info}):
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


class Response(io.BytesIO):
    def __init__(self, body, status=200, range_header=""):
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Range": range_header}


class Downloads(unittest.TestCase):
    def asset(self, body):
        return module.Asset(
            "fixture.zip",
            "https://example.invalid/fixture.zip",
            hashlib.sha256(body).hexdigest(),
            len(body),
            1,
        )

    def test_truncated_response_resumes_and_requires_exact_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            body = b"verified archive"
            requests = []

            def fetch(request, timeout):
                requests.append(request)
                if len(requests) == 1:
                    return Response(body[:5])
                return Response(body[5:], 206, "bytes 5-15/16")

            with patch.object(
                module.urllib.request, "urlopen", side_effect=fetch
            ), patch.object(module.time, "sleep"):
                path = module.download(self.asset(body), Path(directory))
            self.assertEqual(path.read_bytes(), body)
            self.assertEqual(requests[1].get_header("Range"), "bytes=5-")

    def test_server_ignoring_range_restarts_safely(self):
        with tempfile.TemporaryDirectory() as directory:
            body = b"verified archive"
            with patch.object(
                module.urllib.request,
                "urlopen",
                side_effect=[Response(body[:5]), Response(body)],
            ), patch.object(module.time, "sleep"):
                path = module.download(self.asset(body), Path(directory))
            self.assertEqual(path.read_bytes(), body)

    def test_invalid_content_range_cannot_be_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            body = b"verified archive"
            root = Path(directory)
            (root / "fixture.zip.part").write_bytes(body[:5])
            with patch.object(
                module.urllib.request,
                "urlopen",
                side_effect=lambda *args, **kw: Response(
                    body[5:], 206, "bytes 0-10/16"
                ),
            ), patch.object(module.time, "sleep"):
                with self.assertRaises(RuntimeError):
                    module.download(self.asset(body), root)
            self.assertFalse((root / "fixture.zip").exists())

    def test_hash_mismatch_cannot_be_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            body = b"verified archive"
            with patch.object(
                module.urllib.request,
                "urlopen",
                side_effect=lambda *args, **kw: Response(b"x" * len(body)),
            ), patch.object(module.time, "sleep"):
                with self.assertRaises(RuntimeError):
                    module.download(self.asset(body), root)
            self.assertFalse((root / "fixture.zip").exists())
