from src.agents.philosopher_agent import _strip_trailing_citation_inventory


def test_strip_trailing_inventory_single_paragraph_list():
    raw = "Answer body.\n\n[1] \"Plato\", [2] \"Aristotle\", [3] \"Cicero\""
    cleaned = _strip_trailing_citation_inventory(raw)
    assert cleaned == "Answer body."


def test_strip_trailing_inventory_multiline_list():
    raw = "Answer body.\n\n[1] \"Plato\"\n[2] \"Aristotle\"\n[3] \"Cicero\""
    cleaned = _strip_trailing_citation_inventory(raw)
    assert cleaned == "Answer body."


def test_strip_trailing_sources_header_and_list():
    raw = "Answer body.\n\nSources:\n\n[1] \"Plato\", [2] \"Aristotle\""
    cleaned = _strip_trailing_citation_inventory(raw)
    assert cleaned == "Answer body."


def test_strip_trailing_quoted_mapping_list_like_ui_example():
    # Matches the UI screenshot pattern: a trailing block that maps [n] -> quoted title/label.
    raw = (
        "Answer body.\n\n"
        "[1] \"Plato and Aristotle on cycles of civilization,\" "
        "[2] \"Great Depression impacts on global societies,\" "
        "[3] \"Environmental changes in Mesopotamia,\" "
        "[4] \"Cultural shifts in the decline of the Roman Empire,\" "
        "[5] \"Tech in warfare and society destabilization.\""
    )
    cleaned = _strip_trailing_citation_inventory(raw)
    assert cleaned == "Answer body."


def test_does_not_strip_inline_citations_inside_text():
    raw = "Answer body with inline citations [1], [2].\n\nFinal paragraph continues the argument [3]."
    cleaned = _strip_trailing_citation_inventory(raw)
    assert cleaned == raw


def test_does_not_strip_single_bracket_number_paragraph():
    raw = "Answer body.\n\n[1] This is not a bibliography, just a numbered note."
    cleaned = _strip_trailing_citation_inventory(raw)
    assert cleaned == raw


def test_does_not_strip_conclusion_paragraph_with_many_citations():
    # Common pattern: the last paragraph is a conclusion that includes many citations.
    # We should not strip legitimate prose.
    raw = (
        "Answer body.\n\n"
        "In conclusion, the argument turns on providence and agency [1] while also engaging"
        " broader historical contingency [2] and competing ethical frameworks [3], [4], [5]."
    )
    cleaned = _strip_trailing_citation_inventory(raw)
    assert cleaned == raw


def test_does_not_strip_prose_even_if_it_starts_with_citation_token():
    # Rare but possible: a model starts a paragraph with a citation token and then writes prose.
    # Ensure we don't delete it unless it looks like an inventory/mapping block.
    raw = (
        "Answer body.\n\n"
        "[1] In conclusion, this claim is best read as a synthesis rather than a bibliography;"
        " it is defended across multiple sources [2], [3] and reiterated here for emphasis."
    )
    cleaned = _strip_trailing_citation_inventory(raw)
    assert cleaned == raw
