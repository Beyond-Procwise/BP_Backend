"""Incrementally pull one string field out of a JSON document as it streams in.

The Ask model is called with Ollama's ``format: "json"`` and returns
``{"answer": "...", "follow_ups": [...]}``. Streaming its raw tokens straight to
the browser would stream JSON *syntax* — `{"ans`, `wer": "Th`, `e total spe` —
which is precisely the "raw text dump" we do not want on screen. The user must
see prose appear, not a half-built object.

This walks the stream character by character and emits only the decoded contents
of the requested field, with escapes resolved (\\n, \\", \\uXXXX). It never waits
for the document to close, so the first words render as soon as the model has
produced them.
"""

from __future__ import annotations

from typing import Optional

_HEX = "0123456789abcdefABCDEF"


class JsonFieldStreamer:
    """Feed it chunks; it gives back the decoded delta of one top-level field.

    Usage::

        streamer = JsonFieldStreamer("answer")
        for chunk in stream:
            text = streamer.feed(chunk)
            if text:
                emit(text)
    """

    def __init__(self, field: str) -> None:
        self._needle = f'"{field}"'
        self._buffer = ""          # unconsumed input while we hunt for the key
        self._in_value = False     # inside the field's string literal
        self._done = False         # closing quote seen
        self._escape = False       # previous char was a backslash
        self._unicode: Optional[str] = None  # collecting a \uXXXX sequence

    @property
    def complete(self) -> bool:
        return self._done

    def feed(self, chunk: str) -> str:
        """Consume ``chunk``; return whatever new decoded text it contained."""
        if self._done or not chunk:
            return ""

        if not self._in_value:
            self._buffer += chunk
            start = self._locate_value_start()
            if start is None:
                # Keep only enough tail to match a key split across chunk boundaries.
                if len(self._buffer) > len(self._needle) + 8:
                    self._buffer = self._buffer[-(len(self._needle) + 8):]
                return ""
            remainder, self._buffer = self._buffer[start:], ""
            self._in_value = True
            return self._decode(remainder)

        return self._decode(chunk)

    def _locate_value_start(self) -> Optional[int]:
        """Index just past the opening quote of the field's value, if visible yet."""
        key_at = self._buffer.find(self._needle)
        if key_at < 0:
            return None
        cursor = key_at + len(self._needle)
        # Skip whitespace and the colon.
        while cursor < len(self._buffer) and self._buffer[cursor] in " \t\r\n":
            cursor += 1
        if cursor >= len(self._buffer) or self._buffer[cursor] != ":":
            return None
        cursor += 1
        while cursor < len(self._buffer) and self._buffer[cursor] in " \t\r\n":
            cursor += 1
        if cursor >= len(self._buffer):
            return None
        if self._buffer[cursor] != '"':
            # A non-string value (null, a number). Nothing to stream.
            self._done = True
            return None
        return cursor + 1

    def _decode(self, text: str) -> str:
        """Decode JSON string-body characters until the closing quote."""
        out: list[str] = []
        for char in text:
            if self._unicode is not None:
                if char in _HEX:
                    self._unicode += char
                    if len(self._unicode) == 4:
                        try:
                            out.append(chr(int(self._unicode, 16)))
                        except ValueError:
                            pass
                        self._unicode = None
                    continue
                # Malformed escape — abandon it rather than corrupt the output.
                self._unicode = None

            if self._escape:
                self._escape = False
                if char == "u":
                    self._unicode = ""
                else:
                    out.append(
                        {
                            "n": "\n",
                            "t": "\t",
                            "r": "\r",
                            "b": "\b",
                            "f": "\f",
                            '"': '"',
                            "\\": "\\",
                            "/": "/",
                        }.get(char, char)
                    )
                continue

            if char == "\\":
                self._escape = True
                continue

            if char == '"':
                self._done = True
                break

            out.append(char)

        return "".join(out)
