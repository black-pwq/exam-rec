"""Regular-expression based text utilities."""

from __future__ import annotations

import re


class RegexUtil:
    """Reusable regular-expression based text transformations."""

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
