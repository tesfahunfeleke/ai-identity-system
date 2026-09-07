"""
Fact extraction: turns a chunk of raw text into candidate Fact rows.

STATUS: This is a rule-based placeholder, not the real extractor. Phase 3
(LLM Integration) doesn't exist yet, so we can't make real LLM calls here
without duplicating provider-wiring work that belongs in that phase.

The interface below (extract_facts(text) -> list[CandidateFact]) is the
contract the rest of Phase 1B depends on. When Phase 3 ships, replace the
body of extract_facts() with a real LLM call -- nothing in pipeline.py,
facts.py, or the API layer needs to change, since they only depend on this
function's signature.

The current rule-based version recognizes a small set of high-confidence
patterns (e.g. "my favorite X is Y", "I work at X", "I live in X"). It will
miss almost everything else -- that's expected and fine for now. Its real
purpose is to prove the write/supersede pipeline works end-to-end before
Phase 3 makes it actually useful.
"""

import re
from dataclasses import dataclass

# Each pattern: (compiled regex, category, key). The regex's first capture
# group becomes the fact's value.
_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\bmy favorite (\w+(?:\s\w+)?) is ([^.\n]+)", re.IGNORECASE), "preference", None),
    (re.compile(r"\bI work at ([^.\n,]+)", re.IGNORECASE), "trait", "employer"),
    (re.compile(r"\bI live in ([^.\n,]+)", re.IGNORECASE), "trait", "location"),
    (re.compile(r"\bmy (?:partner|spouse|wife|husband) is ([^.\n,]+)", re.IGNORECASE), "relationship", "partner"),
]


@dataclass
class CandidateFact:
    """A fact proposed by the extractor, not yet written to the database."""

    category: str
    key: str
    value: str
    confidence: float
    source_excerpt: str


def extract_facts(text: str) -> list[CandidateFact]:
    """
    Scan a chunk of text for facts using simple regex patterns.

    Confidence is set low (0.3) across the board to reflect that this is a
    crude placeholder -- real LLM-based extraction in Phase 3 should be able
    to assign much higher confidence to well-supported facts.
    """
    candidates: list[CandidateFact] = []

    for pattern, category, fixed_key in _PATTERNS:
        for match in pattern.finditer(text):
            if fixed_key is None:
                # "my favorite X is Y" pattern: key comes from match group 1.
                key = match.group(1).strip().lower().replace(" ", "_")
                value = match.group(2).strip().rstrip(".")
            else:
                key = fixed_key
                value = match.group(1).strip().rstrip(".")

            if not value:
                continue

            candidates.append(
                CandidateFact(
                    category=category,
                    key=key,
                    value=value,
                    confidence=0.3,
                    source_excerpt=match.group(0).strip(),
                )
            )

    return candidates
