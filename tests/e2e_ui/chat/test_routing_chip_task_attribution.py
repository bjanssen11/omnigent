"""Task labels survive the hook, routing relay, live transcript, and reload.

The Claude binary is omitted: realistic Agent inputs go through the real hook
payload builder and server route. Similar tasks must remain attributable even
when their routing rationales overlap.
"""

from __future__ import annotations

import httpx
from playwright.sync_api import Page, expect

from omnigent.inner.hook_scripts.subagent_router import build_route_request
from tests.e2e_ui.conftest import seed_committed_turn

_PARENT_MODEL = "databricks-claude-sonnet-4-6"
_TASKS = ("Review auth.py", "Review sessions.py", "Review tokens.py")


def test_fanout_routing_chips_are_individually_attributable(
    page: Page,
    seeded_session: tuple[str, str],
) -> None:
    """Identify three similar tasks in live chips and persisted raw verdicts."""
    base_url, session_id = seeded_session
    session_url = f"{base_url}/v1/sessions/{session_id}"
    resp = httpx.patch(
        session_url,
        json={"subagent_routing_override": "on"},
        timeout=10.0,
    )
    resp.raise_for_status()
    seed_committed_turn(
        session_id,
        prompt="Fan out three reviewers for auth.py, sessions.py, and tokens.py.",
        reply="Spawning three reviewers now.",
    )

    # Subscribe before routing so the first assertions exercise live delivery.
    with page.expect_response(lambda response: f"{session_url}/stream" in response.url):
        page.goto(f"{base_url}/c/{session_id}")
    expect(page.get_by_test_id("composer-workspace-controls")).to_be_visible()

    for description in _TASKS:
        body = build_route_request(
            {
                "subagent_type": "general-purpose",
                "description": description,
                "prompt": f"{description} for correctness. Report findings without editing.",
            },
            harness="claude-native",
            parent_model=_PARENT_MODEL,
        )
        hook = httpx.post(f"{session_url}/hooks/route-subagent", json=body, timeout=15.0)
        hook.raise_for_status()

    cards = page.get_by_test_id("routing-decision-card")
    expect(cards).to_have_count(3, timeout=15_000)
    for description in _TASKS:
        expect(
            page.get_by_test_id("routing-decision-task").filter(has_text=description)
        ).to_have_text(description)

    items = httpx.get(f"{session_url}/items", timeout=10.0)
    items.raise_for_status()
    decisions = [i for i in items.json()["data"] if i["type"] == "routing_decision"]
    assert sorted(i["task_description"] for i in decisions) == sorted(_TASKS)

    page.reload()
    expect(cards).to_have_count(3, timeout=15_000)
    for description in _TASKS:
        card = cards.filter(
            has=page.get_by_test_id("routing-decision-task").filter(has_text=description)
        )
        expect(card).to_have_count(1)
        expect(card.get_by_test_id("routing-decision-task")).to_have_text(description)
        expect(card.get_by_test_id("routing-decision-scope")).to_contain_text(
            "subagent: general-purpose"
        )
        card.get_by_role("button", name="Show raw routing verdict").click()
        expect(card.locator("pre")).to_contain_text(f'"task_description": "{description}"')
