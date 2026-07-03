"""Regression test for the extraction FileNotFoundError on S3-keyed documents.

process_monitor stores file_path as an S3 object key (e.g.
'documents/Quote/X.pdf'). dispatch_document must resolve it to a local file
(downloading from S3 when not already on disk) before the local-only
FileDetector runs, otherwise extraction fails with FileNotFoundError.
"""
import os
import sys
import tempfile
import types
import unittest

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from src.services.extraction_v3 import dispatch as D


class TestEnsureLocalFile(unittest.TestCase):
    def test_existing_local_file_returned_as_is(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 local")
            path = f.name
        try:
            local, cleanup = D._ensure_local_file(path, agent_nick=None)
            self.assertEqual(local, path)
            self.assertIsNone(cleanup, "local file must not schedule cleanup")
        finally:
            os.remove(path)

    def test_s3_key_downloaded_when_not_local(self):
        key = "documents/Quote/VALUED MERCHANT QUT048597.pdf"
        self.assertFalse(os.path.isfile(key), "precondition: key is not a local path")

        downloaded = {}

        class FakeS3:
            def download_file(self, bucket, k, dest):
                downloaded["bucket"] = bucket
                downloaded["key"] = k
                with open(dest, "wb") as fh:
                    fh.write(b"%PDF-1.4 from-s3")

        agent_nick = types.SimpleNamespace(s3_client=FakeS3())

        # ensure a bucket is configured for the resolver
        D.settings.s3_bucket_name = getattr(D.settings, "s3_bucket_name", None) or "procwisemvp"

        local, cleanup = D._ensure_local_file(key, agent_nick=agent_nick)
        try:
            self.assertTrue(os.path.isfile(local), "downloaded temp file must exist")
            self.assertEqual(cleanup, local, "downloaded temp file must be scheduled for cleanup")
            self.assertEqual(downloaded["key"], key)
            self.assertTrue(local.endswith(".pdf"), "temp file should keep the .pdf suffix")
        finally:
            if os.path.isfile(local):
                os.remove(local)

    def test_missing_everywhere_raises_clear_error(self):
        class FakeS3:
            def download_file(self, *a, **k):
                raise Exception("NoSuchKey")

        agent_nick = types.SimpleNamespace(s3_client=FakeS3())
        with self.assertRaises(FileNotFoundError):
            D._ensure_local_file("definitely/missing.pdf", agent_nick=agent_nick)


if __name__ == "__main__":
    unittest.main()
