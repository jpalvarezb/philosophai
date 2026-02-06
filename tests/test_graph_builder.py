import pandas as pd

from src.graph.builder import GraphBuilder


class DummyStorage:
    def __init__(self, triples_df: pd.DataFrame):
        self._df = triples_df

    def get_triples_df(self) -> pd.DataFrame:
        return self._df


def test_graph_builder_build_coerces_ids_to_strings_and_skips_nulls():
    df = pd.DataFrame(
        [
            {
                "subject_canon_id": 123,
                "predicate_canon_id": 7,
                "object_canon_id": "bond of union",
                "subject_label": "Soul",
                "predicate_label": "serves as",
                "object_label": "bond of union",
                "weight": 2,
                "chunk_ids": ["c1", "c2"],
            },
            # Should be skipped: missing object id
            {
                "subject_canon_id": "soul",
                "predicate_canon_id": "serves_as",
                "object_canon_id": None,
                "subject_label": "soul",
                "predicate_label": "serves as",
                "object_label": None,
                "weight": 1,
                "chunk_ids": [],
            },
            # Should be skipped: empty ids
            {
                "subject_canon_id": "",
                "predicate_canon_id": "x",
                "object_canon_id": "y",
                "subject_label": "",
                "predicate_label": "x",
                "object_label": "y",
                "weight": 1,
                "chunk_ids": [],
            },
        ]
    )

    builder = GraphBuilder(DummyStorage(df))
    G = builder.build()

    # Node IDs are strings
    assert all(isinstance(n, str) for n in G.nodes()), list(G.nodes())

    # Edge endpoints and keys are strings
    for u, v, k in G.edges(keys=True):
        assert isinstance(u, str)
        assert isinstance(v, str)
        assert isinstance(k, str)

    # The valid edge exists
    assert G.has_edge("123", "bond of union", key="7")

    # Skipped rows should not create nodes
    assert "" not in G
    assert None not in G


def test_graph_builder_labels_are_strings():
    df = pd.DataFrame(
        [
            {
                "subject_canon_id": "soul",
                "predicate_canon_id": "serves_as",
                "object_canon_id": "bond of union",
                "subject_label": None,
                "predicate_label": None,
                "object_label": None,
                "weight": 1,
                "chunk_ids": None,
            }
        ]
    )

    builder = GraphBuilder(DummyStorage(df))
    G = builder.build()

    assert G.nodes["soul"]["label"] == "soul"
    assert G.nodes["bond of union"]["label"] == "bond of union"

    # Edge label should default to predicate id
    d = G.get_edge_data("soul", "bond of union")["serves_as"]
    assert isinstance(d.get("label"), str)
    assert d.get("label") == "serves_as"
    assert isinstance(d.get("chunks"), list)
