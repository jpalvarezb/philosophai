import numpy as np

from src.ingest.canonicalizer import EntityCanonicalizer


class DummyStorage:
    con = None


class FakeResponse:
    def __init__(self, content: str):
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": content})()})()]


class FakeCompletions:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self._responses.pop(0))


class FakeLLMClient:
    def __init__(self, responses: list[str]):
        self.chat = type("Chat", (), {"completions": FakeCompletions(responses)})()


def normalized_rows(rows: list[list[float]]) -> np.ndarray:
    arr = np.array(rows, dtype=float)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / norms


def test_cluster_by_similarity_refines_transitive_chain():
    canonicalizer = EntityCanonicalizer(DummyStorage())
    embeddings = normalized_rows(
        [
            [1.0, 0.0, 0.0],
            [0.93, 0.3675, 0.0],
            [0.83, 0.4302, 0.3550],
        ]
    )

    mapping = canonicalizer.cluster_by_similarity(
        ["alpha", "bridge", "gamma"],
        embeddings,
        threshold=0.92,
    )

    assert mapping["alpha"] != mapping["gamma"]


def test_cluster_by_similarity_chooses_centroid_nearest_canonical():
    canonicalizer = EntityCanonicalizer(DummyStorage())
    embeddings = normalized_rows(
        [
            [0.98, 0.2],
            [1.0, 0.0],
            [0.98, -0.2],
        ]
    )

    mapping = canonicalizer.cluster_by_similarity(
        ["EU", "European Union", "EU union"],
        embeddings,
        threshold=0.7,
    )

    assert set(mapping.values()) == {"European Union"}


def test_cluster_by_similarity_forced_pairs_override_blocked_pairs():
    canonicalizer = EntityCanonicalizer(DummyStorage())
    embeddings = normalized_rows(
        [
            [1.0, 0.0],
            [0.4, 0.9165],
        ]
    )
    pair = canonicalizer._pair_key(
        "EU AI Act",
        "European Union Artificial Intelligence Act",
    )

    mapping = canonicalizer.cluster_by_similarity(
        ["EU AI Act", "European Union Artificial Intelligence Act"],
        embeddings,
        threshold=0.9,
        blocked_pairs={pair},
        forced_pairs={pair},
    )

    assert len(set(mapping.values())) == 1
    assert mapping["EU AI Act"] == "European Union Artificial Intelligence Act"


def test_inverse_predicate_guard_detects_active_vs_passive(monkeypatch):
    canonicalizer = EntityCanonicalizer(DummyStorage())

    def fake_core_tokens(predicate: str) -> set[str]:
        if "caus" in predicate:
            return {"cause"}
        if "derive" in predicate or "come" in predicate:
            return {"derive"}
        return set()

    monkeypatch.setattr(canonicalizer, "_predicate_core_tokens", fake_core_tokens)

    assert canonicalizer.is_inverse_predicate_pair("causes", "is caused by")
    assert not canonicalizer.is_inverse_predicate_pair("derived from", "comes from")


def test_preposition_role_guard_blocks_different_roles(monkeypatch):
    canonicalizer = EntityCanonicalizer(DummyStorage())

    def fake_verb_prep(predicate: str):
        p = predicate.lower()
        if "written by" in p or "written by" == p:
            return ("write", "by")
        if "written in" in p or "written in" == p:
            return ("write", "in")
        if "wrote on" in p or "wrote on" == p:
            return ("write", "on")
        if "written about" in p or "written about" == p:
            return ("write", "about")
        return None

    monkeypatch.setattr(canonicalizer, "_get_predicate_verb_preposition", fake_verb_prep)

    assert canonicalizer.has_conflicting_preposition_roles("written by", "written in") is True
    assert canonicalizer.has_conflicting_preposition_roles("written by", "wrote on") is True
    assert canonicalizer.has_conflicting_preposition_roles("written in", "wrote on") is True
    assert canonicalizer.has_conflicting_preposition_roles("written about", "wrote on") is False  # synonym pair
    assert canonicalizer.has_conflicting_preposition_roles("causes", "produces") is False  # no prep


def test_judge_same_entity_pairs_parses_json_batches():
    llm_client = FakeLLMClient(
        [
            (
                '{"pairs":['
                '{"left":"EU AI Act","right":"European Union Artificial Intelligence Act","same_entity":true,"reason":"same law"},'
                '{"left":"Plato","right":"Aristotle","same_entity":false,"reason":"different philosophers"}'
                "]}"
            )
        ]
    )
    canonicalizer = EntityCanonicalizer(DummyStorage(), llm_client=llm_client)

    approved = canonicalizer.judge_same_entity_pairs(
        [
            {"left": "EU AI Act", "right": "European Union Artificial Intelligence Act", "similarity": 0.89},
            {"left": "Plato", "right": "Aristotle", "similarity": 0.89},
        ],
        batch_size=2,
    )

    assert approved == {
        canonicalizer._pair_key(
            "EU AI Act",
            "European Union Artificial Intelligence Act",
        )
    }
