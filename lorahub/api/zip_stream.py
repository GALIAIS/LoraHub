"""Non-seekable write stream for streaming ZIP generation.

When ``zipfile.ZipFile`` is given a seekable file object it seeks back
to patch local file headers with CRC/size after writing each entry.
If the underlying buffer is drained (truncated) between entries the
seek targets stale offsets, causing BytesIO to pad with null bytes and
producing multi-GB corrupt archives from small datasets.

This wrapper reports ``seekable() = False`` so ZipFile uses data
descriptors (ZIP flag bit 3) instead of seeking back.  A monotonic
``tell()`` counter ensures the central directory records correct
absolute offsets.
"""

from __future__ import annotations

import io


class ZipStream(io.RawIOBase):
    """Write-only, non-seekable stream with drainable buffer."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._pos = 0

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    def write(self, data: bytes | bytearray) -> int:  # type: ignore[override]
        n = len(data)
        self._buf.extend(data)
        self._pos += n
        return n

    def tell(self) -> int:
        return self._pos

    def drain(self) -> bytes:
        """Return accumulated bytes and clear the internal buffer."""
        chunk = bytes(self._buf)
        self._buf.clear()
        return chunk
