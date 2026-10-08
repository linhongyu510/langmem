import threading

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel

from langmem import ReflectionExecutor, create_memory_store_manager
from langmem.knowledge import extraction
from langmem.knowledge.extraction import ExtractedMemory, Memory

NAMESPACE = ("memories",)
OLD_VALUE = {"kind": "Memory", "content": {"content": "old preference"}}
NEW_VALUE = {"kind": "Memory", "content": {"content": "new preference"}}
INPUT = {"messages": [HumanMessage(content="Remember my new preference.")]}


class RemoveDoc(BaseModel):
    json_doc_id: str


class FailingStore(InMemoryStore):
    failure = None

    def put(self, namespace, key, value, **kwargs):
        if self.failure is not None:
            raise self.failure
        return super().put(namespace, key, value, **kwargs)

    def delete(self, namespace, key):
        if self.failure is not None:
            raise self.failure
        return super().delete(namespace, key)

    async def aput(self, namespace, key, value, **kwargs):
        return self.put(namespace, key, value, **kwargs)

    async def adelete(self, namespace, key):
        return self.delete(namespace, key)


def make_manager(monkeypatch, store, extract):
    # Replace the model-dependent extraction only; reconciliation, scheduling and
    # persistence run through the real manager and store.
    monkeypatch.setattr(
        extraction,
        "create_memory_manager",
        lambda *args, **kwargs: RunnableLambda(extract),
    )
    return create_memory_store_manager(
        FakeMessagesListChatModel(responses=[AIMessage(content="unused")]),
        store=store,
        namespace=NAMESPACE,
        enable_deletes=True,
    )


def mutation(operation):
    def extract(payload):
        memory_id = "new" if operation == "insert" else payload["existing"][0][0]
        content = (
            RemoveDoc(json_doc_id=memory_id)
            if operation == "delete"
            else Memory(content="new preference")
        )
        return [ExtractedMemory(memory_id, content)]

    return extract


@pytest.mark.anyio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("operation", ["insert", "update", "delete"])
@pytest.mark.parametrize("fail", [False, True], ids=["success", "failure"])
async def test_persistence_outcome_reaches_caller(
    monkeypatch, asynchronous, operation, fail
):
    store = FailingStore()
    if operation != "insert":
        store.put(NAMESPACE, "existing", OLD_VALUE)
    manager = make_manager(monkeypatch, store, mutation(operation))
    error = OSError("store unavailable")
    if fail:
        store.failure = error

    async def run():
        return await manager.ainvoke(INPUT) if asynchronous else manager.invoke(INPUT)

    if fail:
        with pytest.raises(OSError) as caught:
            await run()
        assert caught.value is error
        assert store.get(NAMESPACE, "new") is None
        if operation != "insert":
            assert store.get(NAMESPACE, "existing").value == OLD_VALUE
    else:
        result = await run()
        if operation == "delete":
            assert result == []
            assert store.get(NAMESPACE, "existing") is None
        else:
            key = "new" if operation == "insert" else "existing"
            assert result == [{"namespace": NAMESPACE, "key": key, "value": NEW_VALUE}]
            assert store.get(NAMESPACE, key).value == NEW_VALUE


def test_all_mutations_are_submitted_and_settled_before_failure_returns(monkeypatch):
    delete_finished = threading.Event()
    error = OSError("second write failed")

    class ConcurrentStore(InMemoryStore):
        def put(self, namespace, key, value, **kwargs):
            if key == "first":
                assert delete_finished.wait(5), "delete was not submitted concurrently"
            elif key == "second":
                raise error
            return super().put(namespace, key, value, **kwargs)

        def delete(self, namespace, key):
            super().delete(namespace, key)
            delete_finished.set()

    store = ConcurrentStore()
    store.put(NAMESPACE, "existing", OLD_VALUE)

    def extract(payload):
        return [
            ExtractedMemory("first", Memory(content="new preference")),
            ExtractedMemory("second", Memory(content="new preference")),
            ExtractedMemory(
                payload["existing"][0][0],
                RemoveDoc(json_doc_id=payload["existing"][0][0]),
            ),
        ]

    manager = make_manager(monkeypatch, store, extract)
    with pytest.raises(OSError) as caught:
        manager.invoke(INPUT, config={"max_concurrency": 3})

    assert caught.value is error
    # Successful siblings can persist: failure propagation is not a transaction.
    assert store.get(NAMESPACE, "first").value == NEW_VALUE
    assert store.get(NAMESPACE, "second") is None
    assert store.get(NAMESPACE, "existing") is None


def test_background_reflection_future_reports_persistence_failure(monkeypatch):
    store = FailingStore()
    error = OSError("store unavailable")
    store.failure = error
    manager = make_manager(monkeypatch, store, mutation("insert"))
    executor = ReflectionExecutor(manager, store=store)
    try:
        future = executor.submit(INPUT, {"configurable": {}}, thread_id=None)
        with pytest.raises(OSError) as caught:
            future.result(timeout=5)
        assert caught.value is error
        assert store.get(NAMESPACE, "new") is None
    finally:
        executor.shutdown(wait=False)
