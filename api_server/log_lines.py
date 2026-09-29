import re
from typing import List, Tuple

_LINE_END = re.compile(r'\r\n|\r|\n')


class LogLineSplitter:
    """Splits captured stdout/stderr into lines.

    tqdm redraws its bar in place by writing '\\r' before each frame. Every frame comes out as its own line,
    flagged as a redraw, so progress (and the loss in the bar's postfix) streams as it happens instead of
    piling up into one line until the next newline.
    """

    def __init__(self) -> None:
        self._pending = ''

    def feed(self, chunk: str) -> List[Tuple[str, bool]]:
        """Returns the lines `chunk` completes as `(line, is_redraw)` pairs."""
        self._pending += chunk
        lines: List[Tuple[str, bool]] = []
        start = 0
        for match in _LINE_END.finditer(self._pending):
            # A '\r' ending the buffer may be the first half of a '\r\n' the next chunk completes.
            if match.group() == '\r' and match.end() == len(self._pending):
                break
            line = self._pending[start:match.start()].rstrip()
            start = match.end()
            if line:
                lines.append((line, match.group() == '\r'))
        self._pending = self._pending[start:]
        return lines
