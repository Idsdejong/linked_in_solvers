from __future__ import annotations
import asyncio
import re
from typing import Optional, Tuple, List, Set, Dict
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

import time
import keyboard
import pyautogui

URL = "https://www.linkedin.com/games/zip"

Coord = Tuple[int, int]
Edge = frozenset[Coord]

DEBUG = False
DEBUG_DIR = Path("debug_artifacts")
DEBUG_DIR.mkdir(exist_ok=True)

def _ts():
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")

async def _dump_frames(page, tag="frames"):
    print(f"[DBG] dumping frame tree: {tag}")
    lines = []
    for i, fr in enumerate(page.frames):
        url = fr.url
        try:
            snippet = await fr.evaluate("() => document.body?.innerText?.slice(0,400) || ''")
        except Exception:
            snippet = "(no body)"
        clean_snippet = snippet.replace('\n', ' ')[:200]
        lines.append(f"[{i}] {url}\n  {clean_snippet}…")
    out = "\n".join(lines)
    (DEBUG_DIR / f"{_ts()}_{tag}_frames.txt").write_text(out, encoding="utf-8")
    print(out)

async def _click_any(page_or_frame, labels):
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

# HTML -> grid/barriers parser
def parse_board(html: str):
    soup = BeautifulSoup(html, "lxml")
    board = soup.select_one(".trail-grid.grid-game-board.gil__grid") or \
            soup.select_one(".trail-grid.grid-game-board")
    if board is None:
        raise RuntimeError("Board container not found (selector may need tweaking).")

    style = board.get("style", "")
    rows_match = re.search(r"--rows:\s*(\d+)", style)
    cols_match = re.search(r"--cols:\s*(\d+)", style)
    if not rows_match or not cols_match:
        raise RuntimeError("Could not find --rows/--cols in board style.")
    rows = int(rows_match.group(1))
    cols = int(cols_match.group(1))

    grid: list[list[Optional[int]]] = [[None for _ in range(cols)] for _ in range(rows)]
    barriers: set[frozenset[tuple[int,int]]] = set()

    def add_barrier(a, b):
        if a != b:
            barriers.add(frozenset((a, b)))

    for cell in board.select(".trail-cell"):
        idx_str = cell.get("data-cell-idx")
        if idx_str is None:
            continue
        k = int(idx_str)
        r, c = divmod(k, cols)

        content = cell.select_one(".trail-cell-content")
        if content:
            raw = content.get_text(strip=True)
            if raw and raw.isdigit():
                grid[r][c] = int(raw)

        for wall in cell.select(".trail-cell-wall"):
            classes = " ".join(wall.get("class", []))
            left  = "--left"  in classes
            right = "--right" in classes
            up    = ("--up" in classes) or ("--top" in classes)
            down  = "--down" in classes

            if left  and c-1 >= 0:      add_barrier((r, c), (r, c-1))
            if right and c+1 < cols:    add_barrier((r, c), (r, c+1))
            if up    and r-1 >= 0:      add_barrier((r-1, c), (r, c))
            if down  and r+1 < rows:    add_barrier((r, c), (r+1, c))

    return grid, barriers, rows, cols

# ASCII renderer
def render_ascii(grid, barriers, rows: int, cols: int) -> str:
    def blocked(a, b) -> bool:
        return frozenset((a, b)) in barriers

    H, W = 2*rows + 1, 2*cols + 1
    canvas = [[" " for _ in range(W)] for _ in range(H)]
    for x in range(W):
        canvas[0][x] = "-"
        canvas[H-1][x] = "-"
    for y in range(H):
        canvas[y][0] = "|"
        canvas[y][W-1] = "|"
    canvas[0][0] = canvas[0][W-1] = canvas[H-1][0] = canvas[H-1][W-1] = "+"

    for r in range(rows):
        for c in range(cols):
            cy, cx = 2*r + 1, 2*c + 1
            val = grid[r][c]
            s = "." if val is None else str(val)
            ch = s[-1] if len(s) > 1 else s
            canvas[cy][cx] = ch
            if c+1 < cols and blocked((r,c),(r,c+1)):
                canvas[cy][cx+1] = "│"
            if r+1 < rows and blocked((r,c),(r+1,c)):
                canvas[cy+1][cx] = "─"

    return "\n".join("".join(row) for row in canvas)

# open launcher
# accept cookies (if present)
# click "Start" (button text can vary)
# wait for iframe -> board
# extract board outerHTML and parse
async def fetch_rendered_board_html(url: str) -> str:
    async with async_playwright() as pw:
        # TIP: set headless=False and slow_mo=150 if you want to watch
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(
            locale="en-US",
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0 Safari/537.36"),
        )
        page = await ctx.new_page()
        try:
            print("[STEP] goto launcher")
            await page.goto(url, wait_until="domcontentloaded")
            if DEBUG:
                await page.screenshot(path=DEBUG_DIR / f"{_ts()}_01_after_goto.png", full_page=True)
            await _dump_frames(page, "after_goto")

            # 1) Cookie banner (don’t wait for networkidle; just try a few common labels)
            print("[STEP] try cookie accept")
            cookie_clicked = await _click_any(page, [
                "Accept", "I accept", "Accepteren", "OK",
            ])
            if cookie_clicked:
                print(f"[DBG] cookie clicked via {cookie_clicked}")
            if DEBUG:
                await page.screenshot(path=DEBUG_DIR / f"{_ts()}_02_after_cookie.png", full_page=True)

            # 2) Find launcher iframe (if it exists)
            print("[STEP] find launcher iframe")
            frame = None
            try:
                iframe_el = await page.wait_for_selector("iframe.game-launch-page__iframe", timeout=8000)
                frame = await iframe_el.content_frame()
            except PWTimeout:
                pass
            if frame is None:
                frame = page.main_frame
            await _dump_frames(page, "after_iframe")

            # 3) Click Start/Play (try inside frame first, then page). No networkidle waits.
            print("[STEP] click Start")
            start_label = await _click_any(frame, [
                "Start game", "Start", "Play", "Play now", "Continue", "Begin",
                "Starten", "Doorgaan"
            ]) or await _click_any(page, [
                "Start game", "Start", "Play", "Play now", "Continue", "Begin",
                "Starten", "Doorgaan"
            ])
            if start_label:
                print(f"[DBG] start clicked via {start_label}")
            if DEBUG:
                await page.screenshot(path=DEBUG_DIR / f"{_ts()}_03_after_start.png", full_page=True)

            # 4) Find the board in ANY frame; accept hidden (attached) elements
            print("[STEP] search board in frames")
            board_sel = ".trail-grid.grid-game-board, .trail-grid"
            board_handle = None
            board_frame = None
            # Retry for ~9s total
            for _ in range(30):
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
                await page.wait_for_timeout(300)

            if not board_handle:
                # last attempt: if we had a distinct frame earlier, also try state="attached" there
                try:
                    await frame.wait_for_selector(board_sel, state="attached", timeout=3000)
                    board_handle = await frame.query_selector(board_sel)
                    board_frame = frame if board_handle else None
                except Exception:
                    pass

            if not board_handle:
                await _dump_frames(page, "before_fail")
                # Save HTML of each frame for post-mortem
                for i, fr in enumerate(page.frames):
                    try:
                        html = await fr.content()
                        if DEBUG:
                            (DEBUG_DIR / f"{_ts()}_frame_{i}.html").write_text(html, encoding="utf-8")
                    except Exception:
                        pass
                await page.screenshot(path=DEBUG_DIR / f"{_ts()}_04_fail.png", full_page=True)
                raise RuntimeError("Board not found (likely still behind Start overlay).")

            # Optional: element screenshot (may fail if offscreen/hidden)
            try:
                await board_handle.screenshot(path=DEBUG_DIR / f"{_ts()}_05_board_elem.png")
            except Exception:
                pass

            # 5) Extract just the board’s HTML
            board_html = await board_handle.evaluate("el => el.outerHTML")
            if DEBUG:
                (DEBUG_DIR / f"{_ts()}_board.html").write_text(board_html, encoding="utf-8")
            print("[OK] board HTML extracted")
            return board_html

        finally:
            # always close cleanly
            try:
                if DEBUG:
                    await page.screenshot(path=DEBUG_DIR / f"{_ts()}_99_final.png", full_page=True)
                else:
                    pass
            except Exception:
                pass
            await page.close()
            await ctx.close()
            await browser.close()

def find_number_positions(grid: List[List[Optional[int]]]) -> Dict[int, Coord]:
    pos: Dict[int, Coord] = {}
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if isinstance(v, int):
                pos[v] = (r, c)
    return pos

def solve_zip(grid: List[List[Optional[int]]],
              barriers: Set[Edge],
              rows: int,
              cols: int) -> List[Coord]:
    """
    Returns a path covering all cells (rows*cols) that passes through numbers in order.
    Path is a list of (r, c) coords in visit order.
    Raises ValueError if no solution is found.
    """
    num_cells = rows * cols
    num_pos = find_number_positions(grid)
    if not num_pos:
        raise ValueError("No numbers found on the grid.")

    start_num = min(num_pos.keys())
    end_num   = max(num_pos.keys())
    startLoc  = num_pos[start_num]
    endLoc    = num_pos[end_num]

    # Your DFS, adapted for barriers as frozenset({a,b})
    deltas = [(0,-1),(0,1),(1,0),(-1,0)]
    barriers_set = set(barriers)  # ensure O(1)

    paths: List[List[Coord]] = []  # collect at most 1

    def dfsBackTrack(currNum: int, currPos: Coord, currPath: List[Coord]):
        if currPos == endLoc and len(currPath) == num_cells:
            paths.append(currPath.copy())
            return
        if paths:  # stop on first solution for speed
            return
        r, c = currPos
        for dh, dw in deltas:
            nr, nc = r + dh, c + dw
            newPos = (nr, nc)
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if newPos in currPath:
                continue
            if frozenset({currPos, newPos}) in barriers_set:
                continue
            cell_val = grid[nr][nc]
            if cell_val is not None:
                # must be exactly next numbered checkpoint
                if cell_val == currNum + 1:
                    dfsBackTrack(currNum + 1, newPos, currPath + [newPos])
            else:
                # free cell, continue with the same current number
                dfsBackTrack(currNum, newPos, currPath + [newPos])

    dfsBackTrack(start_num, startLoc, [startLoc])

    if not paths:
        raise ValueError("No valid path found for this board.")
    return paths[0]

def fill_grid_with_flow_numbers(grid: List[List[Optional[int]]],
                                path: List[Coord]) -> List[List[int]]:
    """
    For each step between k and k+1, fill visited cells with k.
    Cells that are exact numbered checkpoints keep their own number.
    """
    # Build a quick lookup for numbered checkpoints
    num_pos = find_number_positions(grid)
    # Inverse map for quick test:
    numbered_cells: Dict[Coord, int] = {pos: k for k, pos in num_pos.items()}

    # We need to know the "current number" as we traverse.
    # Start is min number; whenever we step on the cell for (curr+1), increment.
    if not numbered_cells:
        raise ValueError("Grid has no numbers.")

    curr   = min(num_pos.keys())
    maxnum = max(num_pos.keys())

    out = [[-1 for _ in row] for row in grid]  # will fill with 1..maxnum
    # Set the first cell (must be start)
    r0, c0 = path[0]
    out[r0][c0] = curr  # the start number

    for idx in range(1, len(path)):
        r, c = path[idx]
        if (r, c) in numbered_cells and numbered_cells[(r, c)] == curr + 1:
            curr += 1
        # fill this step’s cell with the current number phase
        out[r][c] = numbered_cells.get((r, c), curr)

    # sanity: replace any -1 (should not happen) with curr
    for r in range(len(out)):
        for c in range(len(out[0])):
            if out[r][c] == -1:
                out[r][c] = curr
    return out

def arrows_from_path(path: List[Coord]) -> str:
    """
    Generate arrow sequence for path steps.
    Up '^', Down 'v', Left '<', Right '>'.
    """
    chars = []
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        dr, dc = r2 - r1, c2 - c1
        if   dr == -1 and dc ==  0: chars.append("^")
        elif dr ==  1 and dc ==  0: chars.append("v")
        elif dr ==  0 and dc == -1: chars.append("<")
        elif dr ==  0 and dc ==  1: chars.append(">")
        else:
            chars.append("?")  # should never happen
    return "".join(chars)

def render_ascii_filled(flow_grid: List[List[int]]) -> str:
    """
    Render the grid with each cell showing its flow number (last digit if >9).
    (No walls here; you already have a wall renderer. This is for the “filled” view.)
    """
    rows, cols = len(flow_grid), len(flow_grid[0]) if flow_grid else 0
    H, W = 2*rows + 1, 2*cols + 1
    canvas = [[" " for _ in range(W)] for _ in range(H)]
    for x in range(W):
        canvas[0][x] = "-"
        canvas[H-1][x] = "-"
    for y in range(H):
        canvas[y][0] = "|"
        canvas[y][W-1] = "|"
    canvas[0][0] = canvas[0][W-1] = canvas[H-1][0] = canvas[H-1][W-1] = "+"

    for r in range(rows):
        for c in range(cols):
            cy, cx = 2*r + 1, 2*c + 1
            val = flow_grid[r][c]
            s = str(val)
            canvas[cy][cx] = s[-1]  # last digit for clarity
    return "\n".join("".join(row) for row in canvas)

def solve_and_render(grid, barriers, rows, cols):
    path = solve_zip(grid, barriers, rows, cols)
    flow = fill_grid_with_flow_numbers(grid, path)
    arrows = arrows_from_path(path)

    print("\n=== Filled flow (numbers show the active segment) ===")
    print(render_ascii_filled(flow))
    print("\n=== Arrow sequence ===")
    print(arrows)
    return path, flow, arrows

def wait_and_play_arrows(arrows: str, delay: float = 0.15):
    print("\nPress 'A' to start executing arrow sequence...")
    keyboard.wait("a")  # blocks until 'a' key is pressed
    print(f"Executing {len(arrows)} arrow presses...")
    for arrow in arrows:
        key = {
            "^": "up",
            "v": "down",
            "<": "left",
            ">": "right",
        }.get(arrow)
        if key:
            pyautogui.press(key)
            time.sleep(delay)
    print("Done executing arrow sequence.")

if __name__ == "__main__":
    # try headless browser to get the live, rendered board
    board_html = asyncio.run(fetch_rendered_board_html(URL))

    # parse and render
    grid, barriers, rows, cols = parse_board(board_html)
    print(f"Board size: {rows} x {cols}")
    print("Grid preview:")
    for r in range(rows):
        print([grid[r][c] for c in range(cols)])

    print(f"\nBarriers: {len(barriers)}")
    print("\nASCII:\n")
    print(render_ascii(grid, barriers, rows, cols))

    path, flow_grid, arrows = solve_and_render(grid, barriers, rows, cols)
    wait_and_play_arrows(arrows, delay=0.001)

