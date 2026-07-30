from exam_rec.utils import RegexUtil


def test_replace_circled() -> None:
    text = (
        r"$ \textcircled{1} $、$\textcircled{10}$、"
        r"$  \textcircled{20}  $"
    )

    assert RegexUtil.replace_circled(text) == "①、⑩、⑳"


def test_replace_circled_preserves_unsupported_numbers() -> None:
    text = r"$ \textcircled{0} $、$ \textcircled{21} $"

    assert RegexUtil.replace_circled(text) == text
