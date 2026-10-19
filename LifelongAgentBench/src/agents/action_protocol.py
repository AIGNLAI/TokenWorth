"""One shared response-to-action protocol for every memory method."""
import re

_ACTIONS = [
    re.compile(r"Action:\s*[A-Za-z_]\w*\(", re.I),
    re.compile(r"Final\s+Answer:\s*(?:Var(?:iable)?\s*)?#\d+", re.I),
    re.compile(r"Action:\s*Operation[\s\S]*?(?:```[\s\S]*?```)", re.I),
    re.compile(r"Action:\s*Answer[\s\S]*", re.I),
    re.compile(r"Act:\s*bash[\s\S]*?(?:```[\s\S]*?```)", re.I),
    re.compile(r"Act:\s*finish[\s\S]*", re.I),
]


def _outside_code_ranges(content: str) -> list[tuple[int, int]]:
    ranges = []
    offset = 0
    in_code = False
    for line in content.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_code = not in_code
        elif not in_code:
            ranges.append((offset, offset + len(line)))
        offset += len(line)
    return ranges


def _call_end(content: str, opening: int) -> int | None:
    depth = 0
    quote = None
    escaped = False
    for index in range(opening, len(content)):
        char = content[index]
        if escaped:
            escaped = False
        elif char == "\\" and quote:
            escaped = True
        elif quote:
            if char == quote:
                quote = None
        elif char in {"'", '"'}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def normalize_first_action(content: str) -> str:
    outside = _outside_code_ranges(content)
    for start, end in outside:
        if re.match(r"\s*user:\s", content[start:end], re.I):
            content = content[:start]
            break
    outside = _outside_code_ranges(content)
    matches = []
    for pattern in _ACTIONS:
        for match in pattern.finditer(content):
            if any(start <= match.start() < end for start, end in outside):
                matches.append(match)
                break
    if not matches:
        return content
    match = min(matches, key=lambda value: value.start())
    if match.group(0).rstrip().endswith("("):
        end = _call_end(content, match.end() - 1)
        # Malformed actions retain their text for the environment validator.
        return content[match.start():end].strip() if end is not None else content[match.start():].strip()
    action = match.group(0).strip()
    if re.match(r"(?:Action:\s*Answer|Act:\s*finish)", action, re.I):
        action = re.split(r"\n\s*(?:Action:|Act:)", action, maxsplit=1, flags=re.I)[0].strip()
    return action
