from __future__ import annotations

import hashlib

from diagnose_level import _sha256_file


def test_checkpoint_sha256_is_recorded_exactly(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"model-checkpoint")

    assert _sha256_file(checkpoint) == hashlib.sha256(
        b"model-checkpoint"
    ).hexdigest()
