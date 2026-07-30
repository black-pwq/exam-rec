"""Regular-expression based text utilities."""

from __future__ import annotations

import re


FULLWIDTH_TO_HALFWIDTH = {
    "\u3000": " ",
    **{
        chr(codepoint): chr(codepoint - 0xFEE0)
        for codepoint in range(0xFF01, 0xFF5F)
    },
}
"""Map fullwidth ASCII forms and the ideographic space to halfwidth text."""


class RegexUtil:
    """Reusable regular-expression based text transformations."""

    FULLWIDTH_TO_HALFWIDTH = FULLWIDTH_TO_HALFWIDTH
    _CIRCLED_PATTERN = re.compile(r"\$\s*\\textcircled\{(\d+)\}\s*\$")

    @staticmethod
    def replace_circled(text: str) -> str:
        """Replace LaTeX ``\\textcircled`` numbers 1 through 20 with Unicode."""

        def replacement(match: re.Match[str]) -> str:
            number = int(match.group(1))
            if 1 <= number <= 20:
                return chr(0x2460 + number - 1)
            return match.group(0)

        return RegexUtil._CIRCLED_PATTERN.sub(replacement, text)
