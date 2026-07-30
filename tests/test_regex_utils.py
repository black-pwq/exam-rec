from exam_rec.utils.regex import FULLWIDTH_TO_HALFWIDTH, RegexUtil


def test_fullwidth_to_halfwidth_contains_ascii_forms_and_space() -> None:
    assert len(FULLWIDTH_TO_HALFWIDTH) == 95
    assert FULLWIDTH_TO_HALFWIDTH["！"] == "!"
    assert FULLWIDTH_TO_HALFWIDTH["Ａ"] == "A"
    assert FULLWIDTH_TO_HALFWIDTH["ｚ"] == "z"
    assert FULLWIDTH_TO_HALFWIDTH["０"] == "0"
    assert FULLWIDTH_TO_HALFWIDTH["～"] == "~"
    assert FULLWIDTH_TO_HALFWIDTH["　"] == " "
    assert RegexUtil.FULLWIDTH_TO_HALFWIDTH is FULLWIDTH_TO_HALFWIDTH
