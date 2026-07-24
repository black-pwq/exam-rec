from collections.abc import Iterator
import json
from types import SimpleNamespace
from typing import Any

import pytest

from extractor.base_extractor import Problem
from extractor.regex_extractor import (
    GeneralRegexExtractor,
    LlmRegexExtractor,
    HuaShengRegexExtractor,
    HuaShengYanyu700RegexExtractor,
    LlmRegexAnalyzer,
    RegexPatternError,
    RegexPatternValidator,
    RegexPatterns,
)
from ocr.base_ocr import BaseOcr, OcrElement, PersistingOcr, Point
from ocr.cached_ocr import CachedOcr


class StubOcr(BaseOcr):
    def __init__(self, *pages: list[OcrElement]) -> None:
        super().__init__()
        self.pages = pages

    def predict_iter(self, input: Any) -> Iterator[list[OcrElement]]:
        yield from self.pages


def result(
    text: str, x_min: float = 200, x_max: float = 800, y_min: float = 10
) -> OcrElement:
    return OcrElement(
        bbox=[
            Point(x_min, y_min),
            Point(x_max, y_min),
            Point(x_max, y_min + 20),
            Point(x_min, y_min + 20),
        ],
        label="text",
        content=text,
    )


def test_extracts_questions_with_multiline_sections() -> None:
    pages = StubOcr(
            [
                result("四海公考"),
                result("1"),
                result("1. （2021年 新疆省考 77%）"),
                result("第一行题干"),
                result("第二行题干"),
                result("A. 选项 A"),
                result("续行"),
                result("B. 选项 B"),
                result("【参考答案】 B"),
                result("【题型分类】 言语理解"),
                result("【实战解析】 解析首行"),
                result("解析续行"),
                result("2. (2020 湖南省考 35%)"),
                result("第二题"),
                result("A. 甲"),
                result("【参考答案】 A"),
            ]
        ).predict_iter(None)
    extractor = HuaShengRegexExtractor()

    assert extractor.extract(pages) == [
        Problem(
            number="1",
            question="第一行题干第二行题干",
            options={"A": "选项 A续行", "B": "选项 B"},
            answer="B",
            analysis="解析首行解析续行",
        ),
        Problem(
            number="2",
            question="第二题",
            options={"A": "甲"},
            answer="A",
            analysis="",
        ),
    ]


def test_appends_right_annotation_column_to_analysis() -> None:
    pages = StubOcr(
            [
                result("1. （2021年 新疆省考 77%）"),
                result("题干公考最新资料、更新进度微信abc123"),
                result("花生批注", 900, 1000, 20),
                result("右侧批注内容", 890, 1050, 30),
                result("A. 正常选项", 100, 700),
            ]
        ).predict_iter(None)
    extractor = HuaShengRegexExtractor()

    assert extractor.extract(pages) == [
        Problem(
            number="1",
            question="题干",
            options={"A": "正常选项"},
            answer="",
            analysis="花生批注右侧批注内容",
        )
    ]


def test_appends_left_annotation_column_to_analysis() -> None:
    pages = StubOcr(
            [
                result("花生批注", 30, 100, 20),
                result("左侧批注", 20, 150, 30),
                result("1. （2021年 新疆省考）", 200, 800, 10),
                result("正常题干", 200, 800, 20),
            ]
        ).predict_iter(None)
    extractor = HuaShengRegexExtractor()

    problem = extractor.extract(pages)[0]
    assert problem.question == "正常题干"
    assert problem.analysis == "花生批注左侧批注"


def test_assigns_annotations_by_problem_y_range() -> None:
    pages = StubOcr(
            [
                result("1. （2021年 新疆省考）", 300, 900, 100),
                result("第一题题干", 300, 900, 140),
                result("①花生批注：", 950, 1050, 180),
                result("第一题批注", 950, 1100, 210),
                result("2. （2020年 湖南省考）", 300, 900, 500),
                result("第二题题干", 300, 900, 540),
                result("②花生批注：", 950, 1050, 600),
                result("第二题批注", 950, 1100, 630),
            ]
        ).predict_iter(None)
    extractor = HuaShengRegexExtractor()

    first, second = extractor.extract(pages)
    assert first.analysis == "①花生批注：第一题批注"
    assert second.analysis == "②花生批注：第二题批注"


def test_continues_problem_across_pages_without_duplicate_output() -> None:
    pages = StubOcr(
            [
                result("1. （2021年 新疆省考）", y_min=100),
                result("第一页", y_min=150),
                result("花生批注", 900, 1000, 180),
                result("跨页批注", 900, 1050, 210),
            ],
            [result("第二页", y_min=100), result("【实战解析】 解析", y_min=200)],
        ).predict_iter(None)
    extractor = HuaShengRegexExtractor()

    assert extractor.extract(pages) == [
        Problem(
            number="1",
            question="第一页第二页",
            options={},
            answer="",
            analysis="解析花生批注跨页批注",
        )
    ]


def test_extracts_yanyu_700_questions_and_options_across_pages() -> None:
    pages = StubOcr(
            [
                result("题源：25 花生十三言语系统班700 词讲义 "),
                result("例题1（2019 年浙江省考） "),
                result("第一题题干第一行"),
                result("第一题题干第二行"),
                result("A．绵延不绝 长盛不衰 交流    B．源远流长 与时俱进 对话"),
                result("C．多姿多彩 历久弥新 解构    D．千姿百态 推陈出新 重构"),
                result("例题2（2018 年联考） "),
                result("第二题第一页题干"),
                result("题本整理：Yummy "),
                result("1"),
            ],
            [
                result("题源：25 花生十三言语系统班700 词讲义 "),
                result("自测用，禁止商用 "),
                result("第二题第二页题干"),
                result("依次填入划横线部分最恰当的一项是："),
                result("A．波澜壮阔 博大精深 历久弥新    B．恢弘壮丽 源远流长 同舟共济"),
                result("C．气势磅礴 奔流不息 与时俱进    D．延绵不断 厚积薄发 奋发有为"),
                result("2"),
            ],
        ).predict_iter(None)
    extractor = HuaShengYanyu700RegexExtractor()

    assert extractor.extract(pages) == [
        Problem(
            number="1",
            question="第一题题干第一行第一题题干第二行",
            options={
                "A": "绵延不绝 长盛不衰 交流",
                "B": "源远流长 与时俱进 对话",
                "C": "多姿多彩 历久弥新 解构",
                "D": "千姿百态 推陈出新 重构",
            },
            answer="",
            analysis="",
        ),
        Problem(
            number="2",
            question="第二题第一页题干第二题第二页题干依次填入划横线部分最恰当的一项是：",
            options={
                "A": "波澜壮阔 博大精深 历久弥新",
                "B": "恢弘壮丽 源远流长 同舟共济",
                "C": "气势磅礴 奔流不息 与时俱进",
                "D": "延绵不断 厚积薄发 奋发有为",
            },
            answer="",
            analysis="",
        ),
    ]


class StubCompletions:
    def __init__(self, content: str | list[str]) -> None:
        self.contents = content if isinstance(content, list) else [content]
        self.request: dict[str, Any] | None = None
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.request = kwargs
        self.requests.append(kwargs)
        index = min(len(self.requests) - 1, len(self.contents) - 1)
        message = SimpleNamespace(content=self.contents[index])
        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=usage,
        )


def llm_client(content: str | list[str]) -> tuple[SimpleNamespace, StubCompletions]:
    completions = StubCompletions(content)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def test_llm_analyzer_uses_supplied_sample_without_requiring_a_match() -> None:
    response = json.dumps(
        {
            "question": r"^Question (?P<number>\d+)$",
            "options": r"(?P<label>[A-D]):(?P<text>.*)$",
            "noise_prefixes": ["第 1 页"],
        }
    )
    client, completions = llm_client(response)
    analyzer = LlmRegexAnalyzer("https://example.test", "key", "model", client)

    patterns = analyzer.analyze_regex_pattern("第1题 题干\nA.甲")

    assert patterns.noise_prefixes == ("第 1 页",)
    assert completions.request is not None
    assert "第1题 题干\nA.甲" in completions.request["messages"][1]["content"]
    assert "extra_body" not in completions.request


def test_llm_analyzer_limits_sample_characters() -> None:
    response = json.dumps(
        {
            "question": r"^Question (?P<number>\d+)$",
            "options": r"(?P<label>[A-D]):(?P<text>.*)$",
        }
    )
    client, completions = llm_client(response)
    analyzer = LlmRegexAnalyzer("https://example.test", "key", "model", client)
    sample = "x" * (analyzer._MAX_SAMPLE_CHARS + 100)

    analyzer.analyze_regex_pattern(sample)

    assert completions.request is not None
    prompt = completions.request["messages"][1]["content"]
    assert prompt == f"<ocr_sample>\n{'x' * analyzer._MAX_SAMPLE_CHARS}\n</ocr_sample>"


def test_llm_analyzer_logs_token_usage_for_each_call(caplog) -> None:
    response = json.dumps(
        {
            "question": r"^第(?P<number>\d+)题$",
            "options": r"(?P<label>[A-D])[.．](?P<text>.*)$",
        }
    )
    client, _ = llm_client(response)
    analyzer = LlmRegexAnalyzer("https://example.test", "key", "model", client)

    with caplog.at_level("INFO", logger="extractor.regex_extractor"):
        analyzer.analyze_regex_pattern("第1题\nA.甲")

    assert "model=model attempt=1" in caplog.text
    assert "prompt_tokens=10 completion_tokens=5 total_tokens=15" in caplog.text


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ("not json", "invalid JSON"),
        (
            json.dumps(
                {
                    "question": r"^第\d+题$",
                    "options": r"(?P<label>[A-D])\.(?P<text>.*)$",
                }
            ),
            "missing groups: number",
        ),
    ],
)
def test_llm_analyzer_rejects_invalid_configuration(
    response: str, message: str
) -> None:
    client, _ = llm_client(response)
    analyzer = LlmRegexAnalyzer("https://example.test", "key", "model", client)

    with pytest.raises(RegexPatternError, match=message):
        analyzer.analyze_regex_pattern("第1题\nA.甲")


def test_llm_analyzer_retries_invalid_named_groups() -> None:
    invalid = json.dumps(
        {
            "question": r"^第(?P<number>\d+)题$",
            "options": r"[A-D][.．].*$",
        }
    )
    valid = json.dumps(
        {
            "question": r"^第(?P<number>\d+)题$",
            "options": r"(?P<label>[A-D])[.．](?P<text>.*)$",
        }
    )
    client, completions = llm_client([invalid, valid])
    analyzer = LlmRegexAnalyzer("https://example.test", "key", "model", client)

    patterns = analyzer.analyze_regex_pattern("第1题\nA.甲")

    assert patterns.options == r"(?P<label>[A-D])[.．](?P<text>.*)$"
    assert len(completions.requests) == 2
    retry_messages = completions.requests[1]["messages"]
    assert retry_messages[-2] == {"role": "assistant", "content": invalid}
    assert "missing groups: label, text" in retry_messages[-1]["content"]


def test_regex_validator_logs_patterns_when_validation_fails(caplog) -> None:
    patterns = RegexPatterns(
        question=r"^第(?P<number>\d+)题$",
        options=r"[A-D][.．].*$",
    )

    with pytest.raises(RegexPatternError, match="missing groups: label, text"):
        RegexPatternValidator.validate(patterns)

    assert "Regex pattern validation failed" in caplog.text
    assert repr(patterns) in caplog.text


@pytest.mark.parametrize(
    "numbered_group",
    ["label1", "label_2", "text1", "text_2"],
)
def test_regex_validator_rejects_numbered_option_groups(numbered_group) -> None:
    patterns = RegexPatterns(
        question=r"^第(?P<number>\d+)题$",
        options=(
            rf"(?P<label>[A-D])[.．](?P<text>.*?)"
            rf"(?P<{numbered_group}>.*)$"
        ),
    )

    with pytest.raises(RegexPatternError, match="must not use numbered groups"):
        RegexPatternValidator.validate(patterns)


def test_general_extractor_handles_configured_layout() -> None:
    patterns = RegexPatterns(
        question=r"^第(?P<number>\d+)题\s*(?P<question>.*)$",
        options=r"(?P<label>[A-D])[.．]\s*(?P<text>.*?)(?=\s+[A-D][.．]|$)",
        answer=r"^答案[:：]\s*(?P<answer>[A-D]+)$",
        noise_prefixes=("页眉",),
    )
    pages = StubOcr(
            [
                result("页眉：模拟试卷"),
                result("第1题 首行"),
                result("题干续行"),
                result("A.甲    B.乙"),
                result("乙的续行"),
                result("答案：B"),
            ]
        ).predict_iter(None)
    extractor = GeneralRegexExtractor(patterns)

    assert extractor.extract(pages) == [
        Problem(
            number="1",
            question="首行题干续行",
            options={"A": "甲", "B": "乙乙的续行"},
            answer="B",
            analysis="",
        )
    ]


def test_llm_extractor_uses_selected_sample_pages_with_one_ocr_pass() -> None:
    class PdfTextOcr(BaseOcr):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def predict_iter(self, input: Any) -> Iterator[list[OcrElement]]:
            self.calls += 1
            for text in ("ignore", "第1题 题干\nA.甲", "第2题 题干\nA.乙"):
                yield [
                    OcrElement(bbox=[], label="text", content=line)
                    for line in text.splitlines()
                ]

    response = json.dumps(
        {
            "question": r"^第(?P<number>\d+)题\s*(?P<question>.*)$",
            "options": r"(?P<label>[A-D])[.．](?P<text>.*)$",
        }
    )
    client, completions = llm_client(response)
    ocr = PdfTextOcr()
    analyzer = LlmRegexAnalyzer("https://example.test", "key", "model", client)
    extractor = LlmRegexExtractor(
        samples_range=range(1, 3),
        analyzer=analyzer,
    )

    assert extractor.extract(ocr.predict_iter(object())) == [
        Problem(
            number="1",
            question="题干",
            options={"A": "甲"},
            answer="",
            analysis="",
        ),
        Problem(
            number="2",
            question="题干",
            options={"A": "乙"},
            answer="",
            analysis="",
        ),
    ]
    assert ocr.calls == 1
    assert completions.request is not None
    sample_prompt = completions.request["messages"][1]["content"]
    assert "第1题 题干" in sample_prompt
    assert "第2题 题干" in sample_prompt
    assert "ignore" not in sample_prompt
    assert extractor.patterns == RegexPatterns(
        question=r"^第(?P<number>\d+)题\s*(?P<question>.*)$",
        options=r"(?P<label>[A-D])[.．](?P<text>.*)$",
    )


def test_llm_extractor_uses_cached_ocr_input(tmp_path) -> None:
    cache = tmp_path / "ocr.jsonl"
    source = StubOcr(
        [result("ignore")],
        [result("第1题 题干"), result("A.甲")],
        [result("第2题 题干"), result("A.乙")],
    )
    PersistingOcr(source, cache).predict(None)
    response = json.dumps(
        {
            "question": r"^第(?P<number>\d+)题\s*(?P<question>.*)$",
            "options": r"(?P<label>[A-D])[.．](?P<text>.*)$",
        }
    )
    client, completions = llm_client(response)
    analyzer = LlmRegexAnalyzer("https://example.test", "key", "model", client)
    extractor = LlmRegexExtractor(
        samples_range=range(1, 2),
        analyzer=analyzer,
    )

    problems = extractor.extract(CachedOcr().predict_iter(cache))

    assert [problem.number for problem in problems] == ["1", "2"]
    assert completions.request is not None
    sample_prompt = completions.request["messages"][1]["content"]
    assert "第1题 题干" in sample_prompt
    assert "第2题 题干" not in sample_prompt
    assert "ignore" not in sample_prompt


def test_llm_extractor_rejects_sample_range_beyond_ocr_pages() -> None:
    response = json.dumps(
        {
            "question": r"^第(?P<number>\d+)题\s*(?P<question>.*)$",
            "options": r"(?P<label>[A-D])[.．](?P<text>.*)$",
        }
    )
    client, _ = llm_client(response)
    extractor = LlmRegexExtractor(
        samples_range=range(0, 2),
        analyzer=LlmRegexAnalyzer("https://example.test", "key", "model", client),
    )

    with pytest.raises(ValueError, match=r"2 > 1"):
        extractor.extract(StubOcr([result("only page")]).predict_iter(None))
