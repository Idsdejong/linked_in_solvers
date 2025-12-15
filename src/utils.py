from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union, List

from playwright.async_api import async_playwright, TimeoutError as PWTimeout


# -----------------------
# Debug / misc utilities
# -----------------------

DEBUG_DIR = Path("debug_artifacts")
DEBUG_DIR.mkdir(exist_ok=True)


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


async def _dump_frames(page, tag: str = "frames") -> None:
    """Debug helper: dump frame tree to a file."""
    lines = []
    for i, fr in enumerate(page.frames):
        url = fr.url
        try:
            snippet = await fr.evaluate(
                "() => document.body?.innerText?.slice(0,400) || ''"
            )
        except Exception:
            snippet = "(no body)"
        clean = snippet.replace("\n", " ")[:200]
        lines.append(f"[{i}] {url}\n  {clean}…")

    out = "\n".join(lines)
    (DEBUG_DIR / f"{_ts()}_{tag}_frames.txt").write_text(out, encoding="utf-8")


async def _click_any(page_or_frame, labels: Sequence[str]) -> Optional[str]:
    """
    Click first matching button with one of the given text labels.
    Returns selector used, or None.
    """
    for txt in labels:
        sel = f"button:has-text('{txt}')"
        try:
            el = await page_or_frame.query_selector(sel)
            if el:
                await el.scroll_into_view_if_needed()
                await el.click(timeout=2000)
                return sel
        except Exception:
            pass
    return None


# -----------------------
# Generic board fetching
# -----------------------

BoardSelector = Union[str, Sequence[str]]


@dataclass(frozen=True)
class FetchOptions:
    """
    Generic options for launching / entering a LinkedIn game.
    Adjust per game if needed, but keep in helper because it's not puzzle-logic.
    """
    headless: bool = True
    locale: str = "en-US"
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )

    # common launcher / overlay selectors
    launcher_iframe_selector: str = "iframe.game-launch-page__iframe"

    # common buttons that appear before the actual board
    cookie_labels: Sequence[str] = ("Accept", "I accept", "Accepteren", "OK")
    start_labels: Sequence[str] = (
        "Start game", "Start", "Play", "Play now", "Continue", "Begin",
        "Starten", "Doorgaan"
    )

    # how long to keep polling for a board (ms total)
    board_search_total_ms: int = 9000
    board_search_step_ms: int = 300


async def fetch_rendered_board_html(
    url: str,
    board_identifier: BoardSelector,
    *,
    options: FetchOptions = FetchOptions(),
    debug: bool = False,
) -> str:
    """
    Game-agnostic:
    - open game url
    - accept cookies if present
    - click Start/Play if present
    - search all frames for element matching board_identifier
    - return element.outerHTML

    board_identifier:
        CSS selector string OR list/tuple of selectors.
        Example for Zip: [".trail-grid.grid-game-board", ".trail-grid"]

    This is the shared stopping point for all games.
    Parsing/solving/rendering is game specific and should live elsewhere.
    """
    if isinstance(board_identifier, str):
        selectors = [board_identifier]
    else:
        selectors = list(board_identifier)

    board_sel = ", ".join(selectors)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=options.headless)
        ctx = await browser.new_context(
            locale=options.locale,
            user_agent=options.user_agent,
        )
        page = await ctx.new_page()

        try:
            # 1) Go to launcher
            await page.goto(url, wait_until="domcontentloaded")
            if debug:
                await page.screenshot(
                    path=DEBUG_DIR / f"{_ts()}_01_after_goto.png",
                    full_page=True
                )
                await _dump_frames(page, "after_goto")

            # 2) Cookies
            await _click_any(page, options.cookie_labels)
            if debug:
                await page.screenshot(
                    path=DEBUG_DIR / f"{_ts()}_02_after_cookie.png",
                    full_page=True
                )

            # 3) Find launcher iframe (if any)
            frame = None
            try:
                iframe_el = await page.wait_for_selector(
                    options.launcher_iframe_selector, timeout=8000
                )
                frame = await iframe_el.content_frame()
            except PWTimeout:
                pass
            if frame is None:
                frame = page.main_frame

            if debug:
                await _dump_frames(page, "after_iframe")

            # 4) Click Start/Play (frame first, then page)
            await (
                _click_any(frame, options.start_labels)
                or _click_any(page, options.start_labels)
            )

            if debug:
                await page.screenshot(
                    path=DEBUG_DIR / f"{_ts()}_03_after_start.png",
                    full_page=True
                )

            # 5) Search for board in ANY frame
            board_handle = None
            board_frame = None
            steps = max(1, options.board_search_total_ms // options.board_search_step_ms)

            for _ in range(steps):
                for fr in page.frames:
                    try:
                        h = await fr.query_selector(board_sel)
                        if h:
                            board_handle = h
                            board_frame = fr
                            break
                    except Exception:
                        pass
                if board_handle:
                    break
                await page.wait_for_timeout(options.board_search_step_ms)

            # last-ditch attempt in the chosen frame
            if not board_handle:
                try:
                    await frame.wait_for_selector(
                        board_sel, state="attached", timeout=3000
                    )
                    board_handle = await frame.query_selector(board_sel)
                    board_frame = frame if board_handle else None
                except Exception:
                    pass

            if not board_handle:
                if debug:
                    await _dump_frames(page, "before_fail")
                    for i, fr in enumerate(page.frames):
                        try:
                            html = await fr.content()
                            (DEBUG_DIR / f"{_ts()}_frame_{i}.html").write_text(
                                html, encoding="utf-8"
                            )
                        except Exception:
                            pass
                    await page.screenshot(
                        path=DEBUG_DIR / f"{_ts()}_04_fail.png",
                        full_page=True
                    )
                raise RuntimeError(
                    f"Board not found using selector(s): {selectors}. "
                    "Likely still behind an overlay or selector needs tweaking."
                )

            if debug:
                try:
                    await board_handle.screenshot(
                        path=DEBUG_DIR / f"{_ts()}_05_board_elem.png"
                    )
                except Exception:
                    pass

            board_html = await board_handle.evaluate("el => el.outerHTML")
            if debug:
                (DEBUG_DIR / f"{_ts()}_board.html").write_text(
                    board_html, encoding="utf-8"
                )
            return board_html

        finally:
            try:
                if debug:
                    await page.screenshot(
                        path=DEBUG_DIR / f"{_ts()}_99_final.png",
                        full_page=True
                    )
            except Exception:
                pass
            await page.close()
            await ctx.close()
            await browser.close()
