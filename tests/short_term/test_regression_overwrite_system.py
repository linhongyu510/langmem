"""Regression tests: provenance of injected summary system messages.

Root cause: the summarizer could not structurally tell an *injected summary* system
message apart from a *user-supplied* system message. It detected the former by a content
substring test (`running_summary.summary in existing_system_message.content`), which
(a) silently dropped the user's own system prompt when the prompt text happened to contain
the summary as a substring, and (b) let a stale injected summary leak into the retained
tail (duplicate summaries).

Fix: track the ids of the summary message(s) this module injects on the persisted
`RunningSummary.generated_message_ids` (state-authenticated message-id provenance) and
recognize them on the next feedback call by that id -- never by content -- so the user
system is always preserved, exactly one summary is emitted, the empty-tail edge still
emits a summary, and the caller's RunningSummary object is never mutated in place.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langmem.short_term.summarization import (
    asummarize_messages,
    summarize_messages,
)
from tests.short_term.utils import FakeChatModel

USER_SYSTEM = "The user's name is Bob. Always address them as Bob."
SUMMARY1 = "The user's name is Bob."


def _seed_messages():
    return [
        SystemMessage(content=USER_SYSTEM, id="sys"),
        HumanMessage(content="hello", id="1"),
        AIMessage(content="hi", id="2"),
        HumanMessage(content="I am Bob", id="3"),
        AIMessage(content="ok Bob", id="4"),
        HumanMessage(content="what's 2+2", id="5"),
        AIMessage(content="4", id="6"),
        HumanMessage(content="latest", id="7"),
    ]


def _with_ids(messages):
    out = []
    for i, m in enumerate(messages):
        m = m.model_copy()
        if m.id is None:
            m.id = f"syn{i}"
        out.append(m)
    return out


def _check(messages):
    sys_msgs = [m for m in messages if m.type == "system"]
    assert any(m.content == USER_SYSTEM for m in sys_msgs), (
        f"user system dropped: {[m.content for m in sys_msgs]}"
    )
    summary_msgs = [
        m
        for m in sys_msgs
        if m.content.startswith("Summary of the conversation so far")
    ]
    assert len(summary_msgs) == 1, (
        f"expected exactly 1 summary system message, got {len(summary_msgs)}: "
        f"{[m.content for m in sys_msgs]}"
    )


def test_overwrite_user_system_not_dropped_substring_sync():
    model = FakeChatModel(responses=[AIMessage(content=SUMMARY1)])
    r1 = summarize_messages(
        _seed_messages(),
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=1,
    )
    # Feed back the overwritten messages + a new turn.
    msgs = _with_ids(list(r1.messages)) + [
        HumanMessage(content="next question", id="8")
    ]
    r2 = summarize_messages(
        msgs,
        running_summary=r1.running_summary,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=1,
    )
    _check(r2.messages)


@pytest.mark.anyio
async def test_overwrite_user_system_not_dropped_substring_async():
    model = FakeChatModel(responses=[AIMessage(content=SUMMARY1)])
    r1 = await asummarize_messages(
        _seed_messages(),
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=1,
    )
    msgs = _with_ids(list(r1.messages)) + [
        HumanMessage(content="next question", id="8")
    ]
    r2 = await asummarize_messages(
        msgs,
        running_summary=r1.running_summary,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=1,
    )
    _check(r2.messages)


def test_empty_tail_running_summary_emits_exactly_one_summary_sync():
    # Scenario E: no new messages, summarized tail already empty.
    from langmem.short_term.summarization import RunningSummary

    rs = RunningSummary(
        summary=SUMMARY1,
        summarized_message_ids={"1", "2"},
        last_summarized_message_id="2",
    )
    model = FakeChatModel(responses=[])
    result = summarize_messages(
        [SystemMessage(content=USER_SYSTEM, id="sys")],
        running_summary=rs,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=1,
    )
    _check(result.messages)


@pytest.mark.anyio
async def test_empty_tail_running_summary_emits_exactly_one_summary_async():
    from langmem.short_term.summarization import RunningSummary

    rs = RunningSummary(
        summary=SUMMARY1,
        summarized_message_ids={"1", "2"},
        last_summarized_message_id="2",
    )
    model = FakeChatModel(responses=[])
    result = await asummarize_messages(
        [SystemMessage(content=USER_SYSTEM, id="sys")],
        running_summary=rs,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=1,
    )
    _check(result.messages)


def test_multi_turn_overwrite_no_duplicate_or_dropped_system_sync():
    model = FakeChatModel(
        responses=[
            AIMessage(content=SUMMARY1),
            AIMessage(content="Extended summary now."),
        ]
    )
    msgs = _seed_messages()
    rs = None
    for _ in range(4):
        r = summarize_messages(
            msgs,
            running_summary=rs,
            model=model,
            token_counter=len,
            max_tokens=6,
            max_summary_tokens=1,
        )
        rs = r.running_summary
        msgs = _with_ids(list(r.messages)) + [
            HumanMessage(content="another", id=f"n{len(msgs)}")
        ]
        _check(r.messages)


def test_callers_running_summary_is_not_mutated_sync():
    """The caller-supplied RunningSummary must not be mutated in place.

    We record a deepcopy of the input RunningSummary, run two distinct feedback calls
    (one that triggers a fresh summary, one that feeds it back), and assert the original
    object is still deep-equal to its snapshot. This guards against in-place
    ``generated_message_ids`` assignment leaking into caller/shared state.
    """
    import copy

    model = FakeChatModel(
        responses=[AIMessage(content=SUMMARY1), AIMessage(content="Extended summary now.")]
    )
    seed = _seed_messages()
    r1 = summarize_messages(
        seed,
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=1,
    )
    # Now reuse the *same* caller RunningSummary object across two different feedback
    # calls and prove it never changes.
    caller_rs = r1.running_summary
    snapshot = copy.deepcopy(caller_rs)

    msgs2 = _with_ids(list(r1.messages)) + [HumanMessage(content="turn two", id="8")]
    summarize_messages(
        msgs2,
        running_summary=caller_rs,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=1,
    )
    msgs3 = _with_ids(list(r1.messages)) + [HumanMessage(content="turn three", id="9")]
    summarize_messages(
        msgs3,
        running_summary=caller_rs,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=1,
    )
    assert caller_rs == snapshot, (
        "caller RunningSummary was mutated in place across feedback calls"
    )


@pytest.mark.anyio
async def test_callers_running_summary_is_not_mutated_async():
    import copy
    model = FakeChatModel(
        responses=[AIMessage(content=SUMMARY1), AIMessage(content="Extended summary now.")]
    )
    r1 = await asummarize_messages(
        _seed_messages(),
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=1,
    )
    caller_rs = r1.running_summary
    snapshot = copy.deepcopy(caller_rs)

    msgs2 = _with_ids(list(r1.messages)) + [HumanMessage(content="turn two", id="8")]
    await asummarize_messages(
        msgs2,
        running_summary=caller_rs,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=1,
    )
    msgs3 = _with_ids(list(r1.messages)) + [HumanMessage(content="turn three", id="9")]
    await asummarize_messages(
        msgs3,
        running_summary=caller_rs,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=1,
    )
    assert caller_rs == snapshot, (
        "caller RunningSummary was mutated in place across async feedback calls"
    )


def test_generated_message_ids_are_state_authenticated():
    """The injected summary is recognized only by the id recorded on RunningSummary.

    After producing a summary, the returned RunningSummary.generated_message_ids must
    contain the summary message's id; feeding the messages back must drop exactly that
    one message and no caller message.
    """
    model = FakeChatModel(responses=[AIMessage(content=SUMMARY1)])
    r1 = summarize_messages(
        _seed_messages(),
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=1,
    )
    gen_ids = r1.running_summary.generated_message_ids
    summary_msgs = [
        m for m in r1.messages if m.content.startswith("Summary of the conversation so far")
    ]
    assert len(summary_msgs) == 1
    assert summary_msgs[0].id in gen_ids
    # Feed back: the old injected summary must be dropped, caller system kept.
    fb = _with_ids(list(r1.messages)) + [HumanMessage(content="next", id="8")]
    r2 = summarize_messages(
        fb,
        running_summary=r1.running_summary,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=1,
    )
    assert all(m.id not in gen_ids for m in r2.messages), (
        "previously injected summary was not dropped by state-authenticated id"
    )
    assert any(m.content == USER_SYSTEM for m in r2.messages)


def test_multiple_template_messages_get_distinct_ids_no_collapse():
    """A custom final_prompt that renders multiple system messages must give each a
    distinct id, so the real add_messages reducer never silently folds two messages.
    """
    from langchain_core.prompts.chat import ChatPromptTemplate

    two_stage = ChatPromptTemplate.from_messages(
        [
            ("placeholder", "{system_message}"),
            ("system", "STATIC RULE THAT MUST SURVIVE"),
            ("system", "Summary of the conversation so far: {summary}"),
            ("placeholder", "{messages}"),
        ]
    )
    model = FakeChatModel(responses=[AIMessage(content=SUMMARY1)])
    r1 = summarize_messages(
        _seed_messages(),
        running_summary=None,
        model=model,
        final_prompt=two_stage,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=1,
    )
    injected = [m for m in r1.messages if m.content in (
        "STATIC RULE THAT MUST SURVIVE",
    ) or m.content.startswith("Summary of the conversation so far")]
    ids = [m.id for m in injected]
    assert len(ids) == 2, f"expected static rule + summary, got {ids}"
    assert len(set(ids)) == 2, f"two template messages share an id and would collapse: {ids}"
    # Both ids are tracked as module-generated.
    assert set(ids) <= r1.running_summary.generated_message_ids
