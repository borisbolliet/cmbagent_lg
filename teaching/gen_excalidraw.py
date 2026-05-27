"""Generate an Excalidraw scene for teaching ReAct + deep_research."""
import json, random, time

random.seed(0)
NOW = int(time.time() * 1000)
elements = []

def _seed():
    return random.randint(1, 2_000_000_000)

def _id():
    return f"id-{random.randint(0, 10**12)}"

def _common(**over):
    base = dict(
        angle=0,
        strokeColor="#1e1e1e",
        backgroundColor="transparent",
        fillStyle="solid",
        strokeWidth=2,
        strokeStyle="solid",
        roughness=1,
        opacity=100,
        groupIds=[],
        frameId=None,
        roundness={"type": 3},
        seed=_seed(),
        version=1,
        versionNonce=_seed(),
        isDeleted=False,
        boundElements=[],
        updated=NOW,
        link=None,
        locked=False,
    )
    base.update(over)
    return base

def rect(x, y, w, h, label=None, fill=None, rounded=True, font_size=20):
    rid = _id()
    over = _common()
    if fill:
        over["backgroundColor"] = fill
        over["fillStyle"] = "solid"
    if not rounded:
        over["roundness"] = None
    rect_el = {"id": rid, "type": "rectangle", "x": x, "y": y, "width": w, "height": h, **over}
    elements.append(rect_el)
    if label:
        tid = _id()
        rect_el["boundElements"].append({"type": "text", "id": tid})
        t = {
            "id": tid, "type": "text",
            "x": x, "y": y, "width": w, "height": h,
            **_common(roundness=None),
            "text": label, "originalText": label,
            "fontSize": font_size, "fontFamily": 5,
            "textAlign": "center", "verticalAlign": "middle",
            "baseline": int(font_size * 0.75),
            "containerId": rid,
            "lineHeight": 1.25,
            "autoResize": True,
        }
        elements.append(t)
    return rid

def ellipse(x, y, w, h, label=None, font_size=18):
    rid = _id()
    over = _common(roundness=None)
    el = {"id": rid, "type": "ellipse", "x": x, "y": y, "width": w, "height": h, **over}
    elements.append(el)
    if label:
        tid = _id()
        el["boundElements"].append({"type": "text", "id": tid})
        t = {
            "id": tid, "type": "text",
            "x": x, "y": y, "width": w, "height": h,
            **_common(roundness=None),
            "text": label, "originalText": label,
            "fontSize": font_size, "fontFamily": 5,
            "textAlign": "center", "verticalAlign": "middle",
            "baseline": int(font_size * 0.75),
            "containerId": rid,
            "lineHeight": 1.25,
            "autoResize": True,
        }
        elements.append(t)
    return rid

def text(x, y, content, font_size=20, align="left"):
    tid = _id()
    width = max(60, int(max(len(line) for line in content.split("\n")) * font_size * 0.55))
    height = int(content.count("\n") + 1) * int(font_size * 1.3)
    elements.append({
        "id": tid, "type": "text",
        "x": x, "y": y, "width": width, "height": height,
        **_common(roundness=None),
        "text": content, "originalText": content,
        "fontSize": font_size, "fontFamily": 5,
        "textAlign": align, "verticalAlign": "top",
        "baseline": int(font_size * 0.75),
        "containerId": None,
        "lineHeight": 1.25,
        "autoResize": True,
    })
    return tid

def arrow(x1, y1, x2, y2, label=None, start=None, end=None):
    aid = _id()
    dx, dy = x2 - x1, y2 - y1
    over = _common(roundness={"type": 2})
    a = {
        "id": aid, "type": "arrow",
        "x": x1, "y": y1, "width": abs(dx), "height": abs(dy),
        **over,
        "points": [[0, 0], [dx, dy]],
        "lastCommittedPoint": None,
        "startBinding": {"elementId": start, "focus": 0, "gap": 4} if start else None,
        "endBinding": {"elementId": end, "focus": 0, "gap": 4} if end else None,
        "startArrowhead": None,
        "endArrowhead": "arrow",
        "elbowed": False,
    }
    elements.append(a)
    if start:
        # mark source's boundElements
        for e in elements:
            if e["id"] == start:
                e["boundElements"].append({"id": aid, "type": "arrow"})
    if end:
        for e in elements:
            if e["id"] == end:
                e["boundElements"].append({"id": aid, "type": "arrow"})
    if label:
        # midpoint text
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        text(int(mx) + 8, int(my) - 14, label, font_size=14, align="left")
    return aid

# ════════════════════ LEFT PANEL: ReAct ════════════════════

text(80, 30, "ReAct loop  (Reason + Act)", font_size=28)
text(80, 70, "From-scratch LangGraph: 2 nodes + 1 conditional edge", font_size=16)

start_id = ellipse(280, 130, 80, 60, "START", font_size=14)
agent_id = rect(220, 250, 200, 90, "agent\n(LLM call)", fill="#e7f5ff")
tools_id = rect(220, 460, 200, 90, "tools\nlookup_population", fill="#fff3bf")
end_id   = ellipse(540, 280, 80, 60, "END", font_size=14)

arrow(320, 190, 320, 250, start=start_id, end=agent_id)
arrow(330, 340, 330, 460, label="tool_calls?", start=agent_id, end=tools_id)
arrow(300, 460, 300, 340, label="observation", start=tools_id, end=agent_id)
arrow(420, 295, 540, 305, label="no tool_calls", start=agent_id, end=end_id)

text(80, 600,
     "think  →  act  →  observe  →  think  →  …\n"
     "The conditional edge IS the loop.\n"
     "Tool result is appended as a ToolMessage,\n"
     "fed back into the next agent call.",
     font_size=16)

# Temperature finding box
rect(80, 760, 580, 170, fill="#f1f3f5")
text(100, 775, "Temperature finding (3 reps × N=10, gemini-3.1-flash-lite)", font_size=18)
text(100, 815,
     "T = 0.0  →  100 %  100 %  100 %   (tool-call rate)\n"
     "T = 1.0  →   50 %   40 %   60 %\n\n"
     "Visible only with a permissive system prompt\n"
     "(\"only call the tool when uncertain\").",
     font_size=15)

# ════════════════════ RIGHT PANEL: deep_research ════════════════════

OFFSET = 820

text(80 + OFFSET, 30, "deep_research  (multi-step orchestration)", font_size=28)
text(80 + OFFSET, 70, "Plan → iterate steps; each step = a fresh self_debug subgraph", font_size=16)

planner_id = rect(180 + OFFSET, 130, 220, 70, "planner ↔ reviewer", fill="#e7f5ff")
plan_id    = rect(180 + OFFSET, 250, 220, 70, "Plan  (N sub-tasks)", fill="#d3f9d8")
step1_id   = rect(120 + OFFSET, 380, 340, 130,
                  "Step 1  · self_debug\nengineer → executor →\nexec_evaluator → step_evaluator",
                  fill="#fff3bf", font_size=16)
step2_id   = rect(120 + OFFSET, 600, 340, 130,
                  "Step 2  · self_debug\nengineer → executor →\nexec_evaluator → step_evaluator",
                  fill="#fff3bf", font_size=16)

arrow(290 + OFFSET, 200, 290 + OFFSET, 250, start=planner_id, end=plan_id)
arrow(290 + OFFSET, 320, 290 + OFFSET, 380, start=plan_id, end=step1_id)
arrow(290 + OFFSET, 510, 290 + OFFSET, 600, start=step1_id, end=step2_id,
      label="previous_steps_execution_summary")

text(80 + OFFSET, 760,
     "Cross-step carryover = the point.\n"
     "Step 2 sees Step 1's code + stdout +\n"
     "workspace file manifest, so it can load\n"
     "what Step 1 wrote.",
     font_size=16)

# Example task
rect(80 + OFFSET, 870, 580, 80, fill="#f8f9fa")
text(100 + OFFSET, 885,
     "Demo task: generate noisy sin(x) → save .npz → load → plot\n"
     "2 steps, 1 attempt each, ~17 s total.",
     font_size=15)

# ════════════════════ wrap up ════════════════════

scene = {
    "type": "excalidraw",
    "version": 2,
    "source": "https://excalidraw.com",
    "elements": elements,
    "appState": {
        "gridSize": None,
        "viewBackgroundColor": "#ffffff",
    },
    "files": {},
}

with open("/tmp/react_and_deep_research.excalidraw", "w") as f:
    json.dump(scene, f, indent=2)

print(f"wrote {len(elements)} elements")
