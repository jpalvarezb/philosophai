"""Export artifact logic: transcript (TXT) with citation renumbering and HTML report."""

from __future__ import annotations

import re
from typing import Any


def _layout_graph_from_trace(
    graph_trace: dict[str, Any],
) -> tuple[list[dict], list[dict]]:
    """
    Build graph layout from graph_trace (used_nodes, used_edges) on the backend.
    Returns (nodes with x, y), (links with source, target, label) for drawing.
    Uses NetworkX layout: kamada_kawai for structure-based positions, fallback spring.
    """
    import networkx as nx

    used_nodes = graph_trace.get("used_nodes") or []
    used_edges = graph_trace.get("used_edges") or []
    if not used_nodes:
        return [], []
    node_list = [n for n in used_nodes if n and n.get("id") is not None]
    if not node_list:
        return [], []
    node_ids = {str(n.get("id")) for n in node_list}
    G = nx.Graph()
    for n in node_list:
        G.add_node(str(n.get("id")), label=n.get("label") or n.get("id"))
    for e in used_edges:
        src = str(e.get("source") or e.get("from") or "")
        tgt = str(e.get("target") or e.get("to") or "")
        if src and tgt and src in node_ids and tgt in node_ids:
            G.add_edge(src, tgt)
    if G.number_of_nodes() == 0:
        return [], []
    try:
        pos = nx.kamada_kawai_layout(G)
    except (nx.NetworkXError, Exception):
        pos = nx.spring_layout(G, seed=42, k=2.5, iterations=100)
    nodes_with_pos = []
    for n in node_list:
        nid = str(n.get("id"))
        if nid in pos:
            x, y = pos[nid]
            nodes_with_pos.append(
                {
                    "id": nid,
                    "label": n.get("label") or nid,
                    "x": float(x),
                    "y": float(y),
                }
            )
    edge_label_map: dict[tuple[str, str], str] = {}
    for e in used_edges:
        src = str(e.get("source") or e.get("from") or "")
        tgt = str(e.get("target") or e.get("to") or "")
        if src in pos and tgt in pos:
            label = e.get("predicate") or e.get("label") or ""
            if isinstance(label, str) and label.strip():
                edge_label_map[(src, tgt)] = label.strip()[:24]
    links = []
    for e in used_edges:
        src = str(e.get("source") or e.get("from") or "")
        tgt = str(e.get("target") or e.get("to") or "")
        if src not in pos or tgt not in pos:
            continue
        link = {"source": src, "target": tgt}
        if (src, tgt) in edge_label_map:
            link["label"] = edge_label_map[(src, tgt)]
        elif (tgt, src) in edge_label_map:
            link["label"] = edge_label_map[(tgt, src)]
        links.append(link)
    return nodes_with_pos, links


def citation_renumber(
    payload: dict[str, Any],
) -> tuple[list[dict], dict[str, int], list[dict]]:
    """
    Deterministic citation renumbering across all PHILO messages.
    Returns: (messages_with_rewritten_content, key_to_index, evidence_entries_ordered).
    """
    key_to_index: dict[str, int] = {}
    next_index = 1
    evidence_entries: list[dict] = []  # { index, citation }

    def citation_key(c: dict) -> str:
        cid = c.get("citation_id") or c.get("chunk_id")
        nid = c.get("node_id")
        if cid and nid:
            return f"{nid}::{cid}"
        return cid or ""

    def assign_index(key: str, citation: dict) -> int:
        nonlocal next_index
        if key in key_to_index:
            return key_to_index[key]
        idx = next_index
        key_to_index[key] = idx
        next_index += 1
        evidence_entries.append({"index": idx, "citation": citation})
        return idx

    out_messages = []
    for msg in payload.get("messages", []):
        role = msg.get("role", "")
        content = msg.get("content", "")
        citations = msg.get("citations") or []

        if role != "philo":
            out_messages.append({**msg, "content": content})
            continue

        # Appearance order: from in-text [n] markers, or fallback to citations[] order
        appearance_order: list[tuple[int, dict]] = []  # (old_1based, citation)
        in_text_refs = re.findall(r"\[(\d+)\]", content)
        if in_text_refs:
            seen_refs: set[int] = set()
            for ref in in_text_refs:
                old_n = int(ref)
                if old_n in seen_refs:
                    continue
                seen_refs.add(old_n)
                # citation with index old_n (1-based)
                c = next(
                    (
                        c
                        for c in citations
                        if c.get("index") == old_n or citations.index(c) + 1 == old_n
                    ),
                    None,
                )
                if not c:
                    c = citations[old_n - 1] if 0 < old_n <= len(citations) else {}
                appearance_order.append((old_n, c))
        else:
            for i, c in enumerate(citations):
                appearance_order.append((i + 1, c))

        old_to_new: dict[int, int] = {}
        for old_n, c in appearance_order:
            key = citation_key(c)
            new_idx = assign_index(key, c)
            old_to_new[old_n] = new_idx

        def replace_ref(match: re.Match) -> str:
            old = int(match.group(1))
            return f"[{old_to_new.get(old, old)}]"

        new_content = re.sub(r"\[(\d+)\]", replace_ref, content)
        out_messages.append({**msg, "content": new_content})

    evidence_entries.sort(key=lambda x: x["index"])
    return out_messages, key_to_index, evidence_entries


def build_transcript_text(payload: dict[str, Any]) -> str:
    """Build plain-text transcript with header, USER/PHILO lines, and evidence appendix."""
    messages, _, evidence_entries = citation_renumber(payload)
    created = payload.get("created_at") or ""
    # Date only: parse ISO and use YYYY-MM-DD
    created_date = created[:10] if created and len(created) >= 10 else created

    lines = [
        "PHILO-001: Transcript",
        created_date,
        "",
        "---",
        "",
    ]

    for msg in messages:
        role = (msg.get("role") or "").upper()
        if role == "USER":
            lines.append("USER:")
        elif role == "PHILO":
            lines.append("PHILO:")
        else:
            lines.append(f"{role}:")
        content = (msg.get("content") or "").strip()
        lines.append(content)
        lines.append("")

    lines.append("")
    lines.append("---")
    lines.append("Evidence appendix")
    lines.append("")

    for entry in evidence_entries:
        idx = entry["index"]
        c = entry["citation"]
        author = c.get("author") or ""
        work = c.get("work_title") or ""
        year = c.get("year") or ""
        node_label = c.get("node_label") or ""
        node_id = c.get("node_id") or ""
        locator = c.get("locator") or c.get("chunk_id") or ""
        excerpt = (c.get("chunk_text") or "").strip()
        year_str = f" ({year})" if year else ""
        lines.append(f"[{idx}] {author} — {work}{year_str}")
        lines.append(f"Node: {node_label} ({node_id})")
        lines.append(f"Locator: {locator}")
        lines.append("Excerpt:")
        lines.append(excerpt)
        lines.append("")

    return "\n".join(lines)


def _escape_html(s: str) -> str:
    """Escape for HTML text content."""
    if not s:
        return ""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _build_graph_only_html(graph_state_json: str) -> str:
    """Build HTML that contains only the graph (same iframe content as report graph section)."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>PHILO-001 Graph</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=VT323&display=swap" rel="stylesheet"/>
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #000; }}
#report-graph {{ width: 100vw; height: 100vh; }}
</style>
</head>
<body>
<div id="report-graph"></div>
<script src="https://cdn.jsdelivr.net/npm/force-graph?v=2"></script>
<script>
(function() {{
  var state = {graph_state_json};
  var nodes = state.nodes || [];
  var links = state.links || [];
  var activeNodeIds = new Set((state.activeNodeIds || state.citedIds || []).map(String));
  var activeLinkKeys = new Set((state.activeLinkKeys || []));
  var viewport = state.viewport || {{ panX: 0, panY: 0, zoom: 1 }};
  if (nodes.length === 0) {{
    document.getElementById('report-graph').innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#666;font-family:sans-serif;">No graph data</div>';
    return;
  }}
  var nodeMap = {{}};
  nodes.forEach(function(n) {{ nodeMap[n.id] = n; n.neighbors = []; }});
  links.forEach(function(l) {{
    var a = nodeMap[l.source], b = nodeMap[l.target];
    if (a && b) {{ a.neighbors.push(b); b.neighbors.push(a); }}
  }});
  var linkRefs = links.map(function(l) {{ return {{ source: nodeMap[l.source], target: nodeMap[l.target], label: l.label || '' }}; }}).filter(function(l) {{ return l.source && l.target; }});
  var Graph = ForceGraph()(document.getElementById('report-graph'))
    .backgroundColor('#000000')
    .nodeCanvasObject(function(node, ctx, globalScale) {{
      ctx.globalAlpha = 1;
      ctx.imageSmoothingEnabled = false;
      var pixelSize = 2, snap = function(v) {{ return Math.round(v / pixelSize) * pixelSize; }};
      var x = snap(node.x), y = snap(node.y);
      var rRaw = Math.max(8, 8 + Math.sqrt(node.val || 0)), r = Math.round(rRaw / pixelSize) * pixelSize;
      var label = node.label || node.id;
      var fontSize = Math.max(9, Math.min(18, r * 2.5)) / globalScale;
      ctx.font = fontSize + 'px "VT323", monospace';
      var isActive = activeNodeIds.has(node.id);
      var strokeColor = isActive ? '#00ff8a' : '#ffffff';
      var limit = r * 1.4, innerR = Math.max(0, r - pixelSize), innerLimit = innerR * 1.4;
      ctx.fillStyle = '#000000';
      if (innerR > 0) {{
        var A = innerR, B = Math.max(0, innerLimit - innerR);
        ctx.beginPath();
        ctx.moveTo(x + A, y + B); ctx.lineTo(x + B, y + A); ctx.lineTo(x - B, y + A); ctx.lineTo(x - A, y + B);
        ctx.lineTo(x - A, y - B); ctx.lineTo(x - B, y - A); ctx.lineTo(x + B, y - A); ctx.lineTo(x + A, y - B);
        ctx.closePath(); ctx.fill();
      }}
      for (var ix = -r; ix <= r; ix += pixelSize) {{
        for (var iy = -r; iy <= r; iy += pixelSize) {{
          var adx = Math.abs(ix), ady = Math.abs(iy), sum = adx + ady;
          if (sum <= limit) {{
            ctx.fillStyle = (adx <= innerR && ady <= innerR && sum <= innerLimit) ? '#000' : strokeColor;
            ctx.fillRect(x + ix, y + iy, pixelSize, pixelSize);
          }}
        }}
      }}
      if (globalScale > 0.8 || isActive) {{
        ctx.fillStyle = isActive ? '#00ff8a' : '#ffffff';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(label, x, y);
      }}
    }})
    .linkCanvasObject(function(link, ctx, globalScale) {{
      ctx.imageSmoothingEnabled = false;
      var snap = function(v) {{ return Math.round(v); }};
      var start = link.source, end = link.target;
      if (typeof start !== 'object' || typeof end !== 'object') return;
      var sx = snap(start.x), sy = snap(start.y), ex = snap(end.x), ey = snap(end.y);
      var dx = ex - sx, dy = ey - sy, dist = Math.sqrt(dx*dx+dy*dy);
      if (dist === 0) return;
      var midX = (sx+ex)/2, midY = (sy+ey)/2, nx = -dy/dist, ny = dx/dist, curvature = 0.2;
      var cpX = snap(midX + nx * dist * curvature), cpY = snap(midY + ny * dist * curvature);
      var linkKey = start.id + '-' + end.id, linkKeyRev = end.id + '-' + start.id;
      var isActive = activeLinkKeys.has(linkKey) || activeLinkKeys.has(linkKeyRev);
      ctx.beginPath(); ctx.moveTo(sx, sy); ctx.quadraticCurveTo(cpX, cpY, ex, ey);
      ctx.strokeStyle = isActive ? '#00ff8a' : 'rgba(255,255,255,0.4)';
      ctx.lineWidth = (isActive ? 2 : 0.5) / globalScale;
      ctx.stroke();
      var label = link.label || link.type || '';
      if (label && (globalScale > 1.2 || isActive)) {{
        var t = 0.5;
        var bx = (1-t)*(1-t)*sx + 2*(1-t)*t*cpX + t*t*ex;
        var by = (1-t)*(1-t)*sy + 2*(1-t)*t*cpY + t*t*ey;
        ctx.save();
        ctx.translate(bx, by); ctx.rotate(Math.atan2(dy, dx));
        ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
        ctx.fillStyle = isActive ? '#00ff8a' : '#ffffff';
        ctx.font = '3px "VT323", monospace';
        ctx.fillText(label, 0, -3);
        ctx.restore();
      }}
    }});
  Graph.graphData({{ nodes: nodes, links: linkRefs }});
  if (typeof Graph.centerAt === 'function' && typeof Graph.zoom === 'function') {{
    Graph.centerAt(viewport.panX || 0, viewport.panY || 0, 0);
    Graph.zoom(viewport.zoom || 1, 0);
  }}
}})();
</script>
</body>
</html>"""


def build_report_html(payload: dict[str, Any]) -> str:
    """Build a self-contained HTML report that embeds the same force-graph as the UI (exact replica)."""
    import json

    graph_trace = payload.get("graph_trace") or {}
    cited_set = set(str(x) for x in (graph_trace.get("cited_node_ids") or []))
    used_nodes = graph_trace.get("used_nodes") or []
    used_edges = graph_trace.get("used_edges") or []
    # Active = exactly what the session shows as green (traversal + cited), no explored vs visited distinction
    active_node_ids = cited_set | {
        str(n.get("id") or n.get("node_id") or "")
        for n in used_nodes
        if n and (n.get("id") or n.get("node_id"))
    }
    active_link_keys: list[str] = []
    for e in used_edges:
        src = str(e.get("source") or e.get("from") or "")
        tgt = str(e.get("target") or e.get("to") or "")
        if src and tgt:
            active_link_keys.append(f"{src}-{tgt}")
            active_link_keys.append(f"{tgt}-{src}")
    edge_label_map: dict[tuple[str, str], str] = {}
    for e in used_edges:
        src = str(e.get("source") or e.get("from") or "")
        tgt = str(e.get("target") or e.get("to") or "")
        label = e.get("predicate") or e.get("label") or ""
        if src and tgt and isinstance(label, str) and label.strip():
            edge_label_map[(src, tgt)] = label.strip()[:24]
            edge_label_map[(tgt, src)] = label.strip()[:24]

    snapshot = payload.get("graph_snapshot")
    if (
        snapshot
        and snapshot.get("nodes")
        and all(
            isinstance(n.get("x"), (int, float))
            and isinstance(n.get("y"), (int, float))
            for n in snapshot["nodes"]
        )
    ):
        nodes = []
        for n in snapshot["nodes"]:
            x, y = float(n["x"]), float(n["y"])
            nodes.append(
                {
                    "id": str(n.get("id") or ""),
                    "label": n.get("label") or n.get("id") or "",
                    "x": x,
                    "y": y,
                    "fx": x,
                    "fy": y,
                    "val": 5,
                }
            )
        node_ids = {n["id"] for n in nodes}
        links = []
        for edge in snapshot.get("links") or []:
            src = str(edge.get("source") or edge.get("from") or "")
            tgt = str(edge.get("target") or edge.get("to") or "")
            if src in node_ids and tgt in node_ids:
                link = {
                    "source": src,
                    "target": tgt,
                    "label": edge.get("label")
                    or edge.get("predicate")
                    or edge_label_map.get((src, tgt))
                    or edge_label_map.get((tgt, src))
                    or "",
                }
                links.append(link)
        viewport = payload.get("graph_viewport") or {"panX": 0, "panY": 0, "zoom": 1}
    else:
        snapshot_nodes, snapshot_links = _layout_graph_from_trace(graph_trace)
        if not snapshot_nodes:
            nodes, links, viewport = [], [], {"panX": 0, "panY": 0, "zoom": 1}
        else:
            nodes = []
            for n in snapshot_nodes:
                x, y = float(n["x"]), float(n["y"])
                nodes.append(
                    {
                        "id": str(n.get("id") or ""),
                        "label": n.get("label") or n.get("id") or "",
                        "x": x,
                        "y": y,
                        "fx": x,
                        "fy": y,
                        "val": 5,
                    }
                )
            links = [
                {
                    "source": str(link.get("source") or link.get("from")),
                    "target": str(link.get("target") or link.get("to")),
                    "label": link.get("label") or link.get("predicate") or "",
                }
                for link in snapshot_links
            ]
            viewport = {"panX": 0, "panY": 0, "zoom": 1}

    messages, _, evidence_entries = citation_renumber(payload)
    created = payload.get("created_at") or ""
    created_date = created[:10] if created and len(created) >= 10 else created
    created_escaped = _escape_html(created_date)
    cited_ids = list(cited_set)

    transcript_html = ""
    for msg in messages:
        role = (msg.get("role") or "").upper()
        content = _escape_html((msg.get("content") or "").strip()).replace(
            "\n", "<br/>"
        )
        transcript_html += f"<p><strong>{role}:</strong> {content}</p>"
    evidence_html = ""
    for entry in evidence_entries:
        idx = entry["index"]
        c = entry["citation"]
        author = _escape_html(str(c.get("author") or ""))
        work = _escape_html(str(c.get("work_title") or ""))
        year = str(c.get("year") or "")
        excerpt = _escape_html((c.get("chunk_text") or "").strip()[:500]).replace(
            "\n", "<br/>"
        )
        year_str = f" ({year})" if year else ""
        evidence_html += f'<div class="citation"><p><strong>[{idx}]</strong> {author} — {work}{year_str}</p><p>{excerpt}</p></div>'

    graph_state_json = json.dumps(
        {
            "nodes": nodes,
            "links": links,
            "citedIds": cited_ids,
            "activeNodeIds": list(active_node_ids),
            "activeLinkKeys": active_link_keys,
            "viewport": viewport,
        }
    )

    embed_only = payload.get("embed_only") is True
    if embed_only:
        return _build_graph_only_html(graph_state_json)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>PHILO-001: Transcript</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=VT323&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet"/>
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: 'Source Serif 4', Georgia, serif; font-size: 1rem; line-height: 1.6; color: #1a1a1a; background: #fafafa; }}
.report {{ max-width: 56rem; margin: 0 auto; padding: 2rem 1.5rem; }}
.report-header {{ border-bottom: 1px solid #e5e5e5; padding-bottom: 1.25rem; margin-bottom: 2rem; }}
.report-header h1 {{ font-size: 1.75rem; font-weight: 600; color: #00aa44; margin: 0 0 0.25rem 0; letter-spacing: 0.02em; }}
.report-header .meta {{ font-size: 0.875rem; color: #666; }}
.report-section {{ margin-bottom: 2.5rem; }}
.report-section h2 {{ font-size: 1.125rem; font-weight: 600; color: #333; margin: 0 0 1rem 0; text-transform: uppercase; letter-spacing: 0.05em; }}
#report-graph {{ width: 100%; height: 56vh; min-height: 400px; background: #000; border-radius: 12px; overflow: hidden; }}
.transcript p {{ margin: 0 0 0.75rem 0; }}
.transcript p:last-child {{ margin-bottom: 0; }}
.evidence p {{ margin: 0 0 0.5rem 0; }}
.evidence .citation {{ margin-bottom: 1.25rem; }}
</style>
</head>
<body>
<div class="report">
<header class="report-header">
<h1>PHILO-001: Transcript</h1>
<p class="meta">{created_escaped}</p>
</header>

<section class="report-section" aria-label="Graph">
<h2>Graph</h2>
<div id="report-graph"></div>
</section>

<section class="report-section" aria-label="Transcript">
<h2>Transcript</h2>
<div class="transcript">{transcript_html}</div>
</section>

<section class="report-section" aria-label="Evidence">
<h2>Evidence appendix</h2>
<div class="evidence">{evidence_html}</div>
</section>
</div>

<script src="https://cdn.jsdelivr.net/npm/force-graph?v=2"></script>
<script>
(function() {{
  var state = {graph_state_json};
  var nodes = state.nodes || [];
  var links = state.links || [];
  var activeNodeIds = new Set((state.activeNodeIds || state.citedIds || []).map(String));
  var activeLinkKeys = new Set((state.activeLinkKeys || []));
  var viewport = state.viewport || {{ panX: 0, panY: 0, zoom: 1 }};
  if (nodes.length === 0) {{
    document.getElementById('report-graph').innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#666;font-family:sans-serif;">No graph data</div>';
    return;
  }}
  var nodeMap = {{}};
  nodes.forEach(function(n) {{ nodeMap[n.id] = n; n.neighbors = []; }});
  links.forEach(function(l) {{
    var a = nodeMap[l.source], b = nodeMap[l.target];
    if (a && b) {{ a.neighbors.push(b); b.neighbors.push(a); }}
  }});
  var linkRefs = links.map(function(l) {{ return {{ source: nodeMap[l.source], target: nodeMap[l.target], label: l.label || '' }}; }}).filter(function(l) {{ return l.source && l.target; }});
  var Graph = ForceGraph()(document.getElementById('report-graph'))
    .backgroundColor('#000000')
    .nodeCanvasObject(function(node, ctx, globalScale) {{
      ctx.globalAlpha = 1;
      ctx.imageSmoothingEnabled = false;
      var pixelSize = 2, snap = function(v) {{ return Math.round(v / pixelSize) * pixelSize; }};
      var x = snap(node.x), y = snap(node.y);
      var rRaw = Math.max(8, 8 + Math.sqrt(node.val || 0)), r = Math.round(rRaw / pixelSize) * pixelSize;
      var label = node.label || node.id;
      var fontSize = Math.max(9, Math.min(18, r * 2.5)) / globalScale;
      ctx.font = fontSize + 'px "VT323", monospace';
      var isActive = activeNodeIds.has(node.id);
      var strokeColor = isActive ? '#00ff8a' : '#ffffff';
      var limit = r * 1.4, innerR = Math.max(0, r - pixelSize), innerLimit = innerR * 1.4;
      ctx.fillStyle = '#000000';
      if (innerR > 0) {{
        var A = innerR, B = Math.max(0, innerLimit - innerR);
        ctx.beginPath();
        ctx.moveTo(x + A, y + B); ctx.lineTo(x + B, y + A); ctx.lineTo(x - B, y + A); ctx.lineTo(x - A, y + B);
        ctx.lineTo(x - A, y - B); ctx.lineTo(x - B, y - A); ctx.lineTo(x + B, y - A); ctx.lineTo(x + A, y - B);
        ctx.closePath(); ctx.fill();
      }}
      for (var ix = -r; ix <= r; ix += pixelSize) {{
        for (var iy = -r; iy <= r; iy += pixelSize) {{
          var adx = Math.abs(ix), ady = Math.abs(iy), sum = adx + ady;
          if (sum <= limit) {{
            ctx.fillStyle = (adx <= innerR && ady <= innerR && sum <= innerLimit) ? '#000' : strokeColor;
            ctx.fillRect(x + ix, y + iy, pixelSize, pixelSize);
          }}
        }}
      }}
      if (globalScale > 0.8 || isActive) {{
        ctx.fillStyle = isActive ? '#00ff8a' : '#ffffff';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(label, x, y);
      }}
    }})
    .linkCanvasObject(function(link, ctx, globalScale) {{
      ctx.imageSmoothingEnabled = false;
      var snap = function(v) {{ return Math.round(v); }};
      var start = link.source, end = link.target;
      if (typeof start !== 'object' || typeof end !== 'object') return;
      var sx = snap(start.x), sy = snap(start.y), ex = snap(end.x), ey = snap(end.y);
      var dx = ex - sx, dy = ey - sy, dist = Math.sqrt(dx*dx+dy*dy);
      if (dist === 0) return;
      var midX = (sx+ex)/2, midY = (sy+ey)/2, nx = -dy/dist, ny = dx/dist, curvature = 0.2;
      var cpX = snap(midX + nx * dist * curvature), cpY = snap(midY + ny * dist * curvature);
      var linkKey = start.id + '-' + end.id, linkKeyRev = end.id + '-' + start.id;
      var isActive = activeLinkKeys.has(linkKey) || activeLinkKeys.has(linkKeyRev);
      ctx.beginPath(); ctx.moveTo(sx, sy); ctx.quadraticCurveTo(cpX, cpY, ex, ey);
      ctx.strokeStyle = isActive ? '#00ff8a' : 'rgba(255,255,255,0.4)';
      ctx.lineWidth = (isActive ? 2 : 0.5) / globalScale;
      ctx.stroke();
      var label = link.label || link.type || '';
      if (label && (globalScale > 1.2 || isActive)) {{
        var t = 0.5;
        var bx = (1-t)*(1-t)*sx + 2*(1-t)*t*cpX + t*t*ex;
        var by = (1-t)*(1-t)*sy + 2*(1-t)*t*cpY + t*t*ey;
        ctx.save();
        ctx.translate(bx, by); ctx.rotate(Math.atan2(dy, dx));
        ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
        ctx.fillStyle = isActive ? '#00ff8a' : '#ffffff';
        ctx.font = '3px "VT323", monospace';
        ctx.fillText(label, 0, -3);
        ctx.restore();
      }}
    }});
  Graph.graphData({{ nodes: nodes, links: linkRefs }});
  if (typeof Graph.centerAt === 'function' && typeof Graph.zoom === 'function') {{
    Graph.centerAt(viewport.panX || 0, viewport.panY || 0, 0);
    Graph.zoom(viewport.zoom || 1, 0);
  }}
}})();
</script>
</body>
</html>"""
    return html
