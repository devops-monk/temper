import json

import pytest

from temper_engine import tools
from temper_engine.tools import Call, expand, extract_calls, flatten, parse_arguments, resolve_name, run

KNOWN = ["calculator", "current_time"]


@pytest.mark.parametrize(
    "raw",
    [
        '{"expression": "1+1"}',
        "{'expression': '1+1'}",  # Python quotes
        '{"expression": "1+1",}',  # trailing comma
        '{"expression": "1+1"',  # cut off
        '{"expression": "1+1',  # cut off inside the string
        '"{\\"expression\\": \\"1+1\\"}"',  # encoded twice
        '```json\n{"expression": "1+1"}\n```',  # fenced
        {"expression": "1+1"},
    ],
)
def test_near_json_arguments_are_repaired(raw):
    assert parse_arguments(raw) == {"expression": "1+1"}


def test_arguments_that_cant_be_read_are_none():
    assert parse_arguments("work out 1+1") is None and parse_arguments('["1+1"]') is None
    assert parse_arguments("") == {} and parse_arguments(None) == {}


def test_nearly_right_tool_names_resolve():
    assert resolve_name("calculator", KNOWN) == "calculator"
    assert resolve_name("functions.Calculator", KNOWN) == "calculator"
    assert resolve_name("current-time", KNOWN) == "current_time"
    assert resolve_name("calculater", KNOWN) == "calculator"
    assert resolve_name("web_search", KNOWN) is None


@pytest.mark.parametrize(
    "text, name, args, left",
    [
        ('Sure.\n<tool_call>\n{"name": "calculator", "arguments": {"expression": "2*3"}}\n</tool_call>', "calculator", {"expression": "2*3"}, "Sure."),
        ('<tool_call>{"name": "calculator", "arguments": {"expression": "2*3"}', "calculator", {"expression": "2*3"}, ""),  # never closed
        ('<|python_tag|>{"name": "calculator", "parameters": {"expression": "2*3"}}<|eom_id|>', "calculator", {"expression": "2*3"}, ""),
        ('[TOOL_CALLS][{"name": "calculator", "arguments": {"expression": "2*3"}}]', "calculator", {"expression": "2*3"}, ""),
        ('<function=calculator>{"expression": "2*3"}</function>', "calculator", '{"expression": "2*3"}', ""),
        ('I will calculate.\n```json\n{"name": "calculator", "arguments": {"expression": "2*3"}}\n```', "calculator", {"expression": "2*3"}, "I will calculate."),
        ('{"name": "calculator", "arguments": {"expression": "2*3"}}', "calculator", {"expression": "2*3"}, ""),
    ],
)
def test_calls_written_into_the_text_are_found_in_each_format(text, name, args, left):
    calls, rest = extract_calls(text, KNOWN)
    assert [(c.name, c.arguments) for c in calls] == [(name, args)] and rest == left


def test_ordinary_text_and_json_are_left_alone():
    assert extract_calls("The answer is 6.", KNOWN) == ([], "The answer is 6.")
    assert extract_calls('{"name": "Ada", "age": 36}', KNOWN) == ([], '{"name": "Ada", "age": 36}')


def test_a_call_that_cant_be_read_goes_back_to_the_model_as_an_error():
    calls, rest = extract_calls("<tool_call>calculator: 2*3</tool_call>", KNOWN)
    step = run(calls[0], KNOWN)
    assert rest == "" and step["error"] and "couldn't be read" in step["result"]


def test_running_a_call_repairs_it_or_says_what_went_wrong():
    assert run(Call("Calculator", "{'expression': '2 ^ 10'}"), KNOWN)["result"] == "2 ** 10 = 1024"
    assert "no tool called 'search'" in run(Call("search", "{}"), KNOWN)["result"]
    assert "weren't valid JSON" in run(Call("calculator", "2+2"), KNOWN)["result"]
    bad = run(Call("calculator", {"expression": "__import__('os').system('ls')"}), KNOWN)
    assert bad["error"] and "can't evaluate" in bad["result"]
    assert run(Call("calculator", {"expression": "9 ** 99999"}), KNOWN)["error"]


def test_the_calculator_and_clock():
    assert tools.calculator({"expression": "sqrt(16) + 10 % 4"}) == "sqrt(16) + 10 % 4 = 6"
    assert tools.calculator({"expression": "1/3"}).startswith("1/3 = 0.333")
    assert "Asia/Kolkata" in tools.current_time({"timezone": "Asia/Kolkata"}) and "UTC+05:30" in tools.current_time({"timezone": "Asia/Kolkata"})
    with pytest.raises(ValueError):
        tools.current_time({"timezone": "Mars/Olympus"})


def test_saved_steps_become_the_rounds_the_model_saw():
    reply = {"role": "assistant", "content": "Checking.Done: 4.", "tools": [{"id": "a", "name": "calculator", "arguments": {"expression": "2+2"}, "result": "2+2 = 4", "error": False, "at": 9}]}
    native = expand([{"role": "user", "content": "2+2?"}, reply], native=True)
    assert [m["role"] for m in native] == ["user", "assistant", "tool", "assistant"]
    assert native[1]["content"] == "Checking." and json.loads(native[1]["tool_calls"][0]["function"]["arguments"]) == {"expression": "2+2"}
    assert native[2]["content"] == "2+2 = 4" and native[3] == {"role": "assistant", "content": "Done: 4."}
    prompted = expand([reply], native=False)
    assert [m["role"] for m in prompted] == ["assistant", "user", "assistant"] and "<tool_call>" in prompted[0]["content"] and "<tool_response" in prompted[1]["content"]
    # Counting sees the native rounds as alternating text turns.
    flat = flatten(native)
    assert [m["role"] for m in flat] == ["user", "assistant", "user", "assistant"] and "<tool_call>" in flat[1]["content"]


def test_the_prompt_offers_only_the_named_tools():
    text = tools.prompt(["calculator"])
    assert "calculator" in text and "current_time" not in text and "<tool_call>" in text
    assert tools.prompt([]) == "" and tools.schemas(["nope"]) == []
