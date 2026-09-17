"""E2E: rail auto-alignment must not page history or move the transcript bottom."""

from __future__ import annotations

from playwright.sync_api import Page, Request, expect

from tests.e2e_ui.chat.test_transcript_scroll_persistence import (
    _BOTTOM_DISTANCE,
    _SCROLL_TO_BOTTOM,
    _TURNS,
    _seed_turns,
)

_AT_BOTTOM_PX = 8
_SETTLE_WINDOW_MS = 4_000
_READ_INTERVAL_MS = 250
_STABLE_READS = 8


def _track_history_paging(page: Page) -> list[str]:
    """Collect item fetches that page older history (carry an `after` cursor)."""
    paging: list[str] = []

    def on_request(request: Request) -> None:
        if "/items?" in request.url and "after=" in request.url:
            paging.append(request.url)

    page.on("request", on_request)
    return paging


def _open_seeded_pair(
    page: Page,
    seeded_session_pair: tuple[str, str, str],
) -> tuple[str, str, str]:
    base_url, session_a, session_b = seeded_session_pair
    _seed_turns(session_a, "alpha")
    _seed_turns(session_b, "beta")
    page.set_viewport_size({"width": 1280, "height": 600})
    page.goto(f"{base_url}/c/{session_a}")
    expect(page.get_by_text(f"alpha reply {_TURNS - 1}").first).to_be_visible(timeout=30_000)
    expect(page.locator(f'a[href="/c/{session_b}"]')).to_be_visible(timeout=30_000)
    return base_url, session_a, session_b


def _resting_bottom_distance(page: Page) -> float | None:
    """Bottom distance once history churn settles (no loading indicator, ~2s unchanged)."""
    indicator = page.get_by_text("Loading earlier messages")
    last: float | None = None
    stable = 0
    for _ in range(100):
        page.wait_for_timeout(_READ_INTERVAL_MS)
        if indicator.count():
            last, stable = None, 0
            continue
        distance = page.evaluate(_BOTTOM_DISTANCE)
        if distance is not None and last is not None and abs(distance - last) < 1:
            stable += 1
            if stable >= _STABLE_READS:
                return distance
        else:
            stable = 0
        last = distance
    return last


def test_opening_long_transcript_does_not_page_history(
    page: Page,
    seeded_session_pair: tuple[str, str, str],
) -> None:
    """Without a reader gesture, opening a transcript fetches no older history."""
    base_url, _session_a, session_b = _open_seeded_pair(page, seeded_session_pair)
    paging = _track_history_paging(page)

    page.locator(f'a[href="/c/{session_b}"]').click()
    expect(page).to_have_url(f"{base_url}/c/{session_b}", timeout=15_000)
    expect(page.get_by_text(f"beta reply {_TURNS - 1}").first).to_be_visible(timeout=30_000)
    page.wait_for_timeout(_SETTLE_WINDOW_MS)

    assert paging == []


def test_bottom_rests_at_bottom_after_conversation_switch(
    page: Page,
    seeded_session_pair: tuple[str, str, str],
) -> None:
    """A transcript left at bottom still rests there once history churn settles."""
    base_url, session_a, session_b = _open_seeded_pair(page, seeded_session_pair)
    page.evaluate(_SCROLL_TO_BOTTOM)

    page.locator(f'a[href="/c/{session_b}"]').click()
    expect(page).to_have_url(f"{base_url}/c/{session_b}", timeout=15_000)
    expect(page.get_by_text(f"beta reply {_TURNS - 1}").first).to_be_visible(timeout=30_000)
    page.locator(f'a[href="/c/{session_a}"]').click()
    expect(page).to_have_url(f"{base_url}/c/{session_a}", timeout=15_000)
    expect(page.get_by_text(f"alpha reply {_TURNS - 1}").first).to_be_visible(timeout=30_000)

    distance = _resting_bottom_distance(page)
    assert distance is not None and distance <= _AT_BOTTOM_PX, distance
