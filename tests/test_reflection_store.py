from typing import Any

import pytest
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph._internal._constants import CONFIG_KEY_RUNTIME
from langgraph.config import get_store
from langgraph.constants import CONF
from langgraph.func import entrypoint
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore

from langmem import ReflectionExecutor
from langmem.utils import NamespaceTemplate


class NoOpReflector(Runnable):
    namespace = NamespaceTemplate(("memories",))

    def invoke(self, input: dict, config: RunnableConfig | None = None, **kwargs: Any):
        return input


class StoreWriter(NoOpReflector):
    def invoke(self, input: dict, config: RunnableConfig | None = None, **kwargs: Any):
        store = get_store()
        store.put(self.namespace(), input["key"], input["value"])
        return input["key"]


def test_background_reflection_retains_runtime_store():
    store = InMemoryStore()
    with ReflectionExecutor(StoreWriter()) as executor:

        @entrypoint(store=store)
        def reflect(payload: dict):
            return executor.submit(payload, thread_id=None).result(timeout=5)

        assert (
            reflect.invoke({"key": "first", "value": {"text": "likes cats"}}) == "first"
        )
        assert store.get(("memories",), "first").value == {"text": "likes cats"}

        # Once resolved, the store also serves submissions outside a graph.
        future = executor.submit(
            {"key": "second", "value": {"text": "likes dogs"}},
            {CONF: {}},
            thread_id=None,
        )
        assert future.result(timeout=5) == "second"
        assert store.get(("memories",), "second").value == {"text": "likes dogs"}


@pytest.mark.anyio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["search", "asearch"])
async def test_search_uses_store_resolved_by_submit(asynchronous):
    store = InMemoryStore()
    store.put(("memories",), "known", {"text": "likes cats"})
    with ReflectionExecutor(NoOpReflector()) as executor:

        @entrypoint(store=store)
        def reflect(payload: dict):
            return executor.submit(payload, thread_id=None).result(timeout=5)

        assert reflect.invoke({}) == {}
        results = await executor.asearch() if asynchronous else executor.search()
        assert [(item["key"], item["value"]) for item in results] == [
            ("known", {"text": "likes cats"})
        ]


@pytest.mark.parametrize("with_runtime", [False, True])
def test_explicit_store_takes_precedence(with_runtime):
    explicit_store = InMemoryStore()
    runtime_store = InMemoryStore()
    config = {
        CONF: {CONFIG_KEY_RUNTIME: Runtime(store=runtime_store)} if with_runtime else {}
    }
    with ReflectionExecutor(StoreWriter(), store=explicit_store) as executor:
        future = executor.submit(
            {"key": "known", "value": {"text": "likes cats"}},
            config,
            thread_id=None,
        )
        assert future.result(timeout=5) == "known"
        assert explicit_store.get(("memories",), "known").value == {
            "text": "likes cats"
        }
        assert runtime_store.get(("memories",), "known") is None


@pytest.mark.parametrize("with_runtime", [False, True], ids=["absent", "empty"])
def test_missing_store_rejection_allows_later_resolution(with_runtime):
    config = {CONF: {CONFIG_KEY_RUNTIME: Runtime()} if with_runtime else {}}
    store = InMemoryStore()
    with ReflectionExecutor(StoreWriter()) as executor:
        with pytest.raises(ValueError, match="could not resolve store"):
            executor.submit(
                {"key": "rejected", "value": {"text": "must not be written"}},
                config,
                thread_id=None,
            )

        @entrypoint(store=store)
        def reflect(payload: dict):
            return executor.submit(payload, thread_id=None).result(timeout=5)

        assert (
            reflect.invoke({"key": "known", "value": {"text": "likes cats"}}) == "known"
        )
        assert store.get(("memories",), "known").value == {"text": "likes cats"}
        assert store.get(("memories",), "rejected") is None
