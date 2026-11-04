"""Display findings already present in a retrieved abstract, without inference."""
from __future__ import annotations

import re


HEADINGS = re.compile(
    r"\b(?P<label>(?:main\s+)?results|findings|conclusions?|interpretation|"
    r"discussion|funding|trial\s+registration|keywords)\s*:", re.I)


def excerpt(text, limit):
    text = text.strip()
    if len(text) <= limit:
        return text, False
    cut = text.rfind(". ", 0, limit)
    if cut < limit // 2:
        cut = text.rfind(" ", 0, limit)
    else:
        cut += 1
    return text[:max(1, cut)].strip(), True


def source_sections(source):
    """Return literal source excerpts; no model calls or numerical valuation.

    Some abstracts have no headings. For those, display source text as a whole
    rather than guessing which sentences report an outcome. Missing results or
    interval estimates never become a claim that an intervention had no effect.
    """
    if not source.get("abstract_available") or not source.get("text", "").strip():
        return []
    text = source["text"]
    matches = list(HEADINGS.finditer(text))
    sections = []
    for i, match in enumerate(matches):
        label = match["label"].casefold()
        if label in ("results", "main results", "findings"):
            title, limit = "Reported results — verbatim abstract", 1800
        elif label in ("conclusion", "conclusions", "interpretation"):
            title, limit = "Authors' interpretation — verbatim abstract", 1000
        else:
            continue
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        quote, truncated = excerpt(text[match.end():end], limit)
        if quote:
            sections.append(dict(label=title, quote=quote, truncated=truncated))
    if not any(s["label"].startswith("Reported results") for s in sections):
        # Include the full unsectioned text; conclusions alone must not be
        # presented as if they were the measured results.
        quote, truncated = excerpt(text, 2600)
        sections.insert(0, dict(label="Retrieved abstract — inspect reported findings", quote=quote, truncated=truncated))
    return sections
