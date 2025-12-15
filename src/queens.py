from __future__ import annotations
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Dict, Set
from bs4 import BeautifulSoup
import asyncio

from utils import fetch_rendered_board_html, FetchOptions

Coord = Tuple[int, int]

class CellState(Enum):
    EMPTY = 0
    QUEEN = 1
    X = 2   # “blocked” mark by the player / given state

@dataclass(frozen=True)
class QueensBoard:
    rows: int
    cols: int
    regions: List[List[int]]               # region id per cell (from cell-color-N)
    state: List[List[CellState]]           # EMPTY / QUEEN / X
    given_queens: Set[Coord]               # queens that are fixed at start (aria-disabled=true)
    given_x: Set[Coord]                    # if the game ever gives X’s (rare, but supported)

    @property
    def n(self) -> int:
        return self.rows * self.cols


def parse_queens_board(html: str) -> QueensBoard:
    soup = BeautifulSoup(html, "lxml")

    grid_el = soup.select_one("#queens-grid") \
           or soup.select_one(".queens-grid-no-gap") \
           or soup.select_one(".queens-grid")

    if grid_el is None:
        raise RuntimeError("Queens grid container not found. Selector may need tweaking.")

    style = grid_el.get("style", "")
    rows_m = re.search(r"--rows:\s*(\d+)", style)
    cols_m = re.search(r"--cols:\s*(\d+)", style)
    if not rows_m or not cols_m:
        raise RuntimeError("Could not find --rows/--cols in queens grid style.")

    rows = int(rows_m.group(1))
    cols = int(cols_m.group(1))

    regions: List[List[int]] = [[-1 for _ in range(cols)] for _ in range(rows)]
    state: List[List[CellState]] = [[CellState.EMPTY for _ in range(cols)] for _ in range(rows)]
    given_queens: Set[Coord] = set()
    given_x: Set[Coord] = set()

    cell_els = grid_el.select(".queens-cell-with-border")
    if not cell_els:
        cell_els = grid_el.select("[data-cell-idx]")

    color_re = re.compile(r"\bcell-color-(\d+)\b")

    for cell in cell_els:
        idx_str = cell.get("data-cell-idx")
        if idx_str is None:
            continue
        idx = int(idx_str)
        r, c = divmod(idx, cols)

        classes = " ".join(cell.get("class", []))
        m = color_re.search(classes)
        if m:
            region_id = int(m.group(1))
        else:
            raise RuntimeError(f"Region id not found for cell idx={idx}. Classes={classes!r}")

        regions[r][c] = region_id

        aria_label = (cell.get("aria-label") or "").lower()
        aria_disabled = (cell.get("aria-disabled") or "").lower() == "true"

        has_queen_dom = cell.select_one(".cell-input--queen") is not None
        has_x_dom = cell.select_one(".cell-input--x, .cell-input--cross, .cell-input--blocked") is not None

        has_queen_aria = "queen of color" in aria_label or "queen," in aria_label
        has_x_aria = "x of color" in aria_label or "blocked cell" in aria_label or "marked x" in aria_label

        if has_queen_dom or has_queen_aria:
            state[r][c] = CellState.QUEEN
            if aria_disabled:
                given_queens.add((r, c))
        elif has_x_dom or has_x_aria:
            state[r][c] = CellState.X
            if aria_disabled:
                given_x.add((r, c))
        else:
            state[r][c] = CellState.EMPTY

    if any(regions[r][c] == -1 for r in range(rows) for c in range(cols)):
        raise RuntimeError("Some regions were not parsed; selector mismatch.")

    return QueensBoard(
        rows=rows,
        cols=cols,
        regions=regions,
        state=state,
        given_queens=given_queens,
        given_x=given_x,
    )


def render_queens_ascii(
    board: QueensBoard,
    show_regions: bool = False,
    solution_queens: Optional[Set[Coord]] = None,
) -> str:
    """
    ASCII render of Queens board.

    Cells:
      Q = queen (given or solution)
      x = X mark (blocked by given/player)
      . = empty

    If show_regions=True, prints region ids instead of pieces.
    If solution_queens provided, overlays those queens.
    """
    rows, cols = board.rows, board.cols
    solution_queens = solution_queens or set()

    lines = []
    lines.append(f"Queens board: {rows} x {cols}")
    if show_regions:
        lines.append("Legend: numbers = region/color id")
    else:
        lines.append("Legend: Q=queen, x=blocked, .=empty")

    col_header = "    " + " ".join(f"{c+1:>2}" for c in range(cols))
    lines.append(col_header)

    top = "   +" + "---"*cols + "+"
    lines.append(top)

    for r in range(rows):
        row_cells = []
        for c in range(cols):
            if show_regions:
                row_cells.append(f"{board.regions[r][c]:>2}")
            else:
                if (r, c) in solution_queens or board.state[r][c] == CellState.QUEEN:
                    row_cells.append(" Q")
                elif board.state[r][c] == CellState.X:
                    row_cells.append(" x")
                else:
                    row_cells.append(" .")
        lines.append(f"{r+1:>2} |" + "".join(row_cells) + " |")

    lines.append(top)
    return "\n".join(lines)


# -----------------------------
# Solver: propagate + MRV + LCV
# -----------------------------

def solve_queens(board: QueensBoard) -> Set[Coord]:
    rows, cols = board.rows, board.cols
    region_id = board.regions

    # forbidden cells from X marks
    forbidden: Set[Coord] = {
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if board.state[r][c] == CellState.X
    }

    # region -> cells
    region_cells: Dict[int, List[Coord]] = {}
    for r in range(rows):
        for c in range(cols):
            k = region_id[r][c]
            region_cells.setdefault(k, []).append((r, c))
    all_regions = set(region_cells.keys())

    # neighbors (king adjacency, incl diagonals)
    neighbors: Dict[Coord, List[Coord]] = {}
    for r in range(rows):
        for c in range(cols):
            nb = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        nb.append((nr, nc))
            neighbors[(r, c)] = nb

    # mutable state
    queens: Set[Coord] = set()
    used_cols: Set[int] = set()
    used_regions: Set[int] = set()
    row_filled: List[bool] = [False] * rows

    # adjacency blocks with ref-counting
    blocked_counts: Dict[Coord, int] = {}

    def block_cell(cell: Coord):
        blocked_counts[cell] = blocked_counts.get(cell, 0) + 1

    def unblock_cell(cell: Coord):
        cnt = blocked_counts.get(cell, 0)
        if cnt <= 1:
            blocked_counts.pop(cell, None)
        else:
            blocked_counts[cell] = cnt - 1

    def is_blocked(cell: Coord) -> bool:
        return blocked_counts.get(cell, 0) > 0

    def conflicts_with_existing(cell: Coord) -> bool:
        r, c = cell
        k = region_id[r][c]
        if row_filled[r] or c in used_cols or k in used_regions:
            return True
        if is_blocked(cell):
            return True
        return False

    def place(cell: Coord):
        r, c = cell
        k = region_id[r][c]

        queens.add(cell)
        used_cols.add(c)
        used_regions.add(k)
        row_filled[r] = True

        newly_blocked: List[Coord] = []
        for nb in neighbors[cell]:
            if nb not in queens:
                before = blocked_counts.get(nb, 0)
                block_cell(nb)
                if before == 0:
                    newly_blocked.append(nb)
        return newly_blocked

    def unplace(cell: Coord, newly_blocked: List[Coord]):
        r, c = cell
        k = region_id[r][c]

        queens.remove(cell)
        used_cols.remove(c)
        used_regions.remove(k)
        row_filled[r] = False

        # decrement all neighbor blocks
        for nb in neighbors[cell]:
            if nb not in queens:
                unblock_cell(nb)

    # init from given queens (including queens already on board)
    given = set(board.given_queens) | {
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if board.state[r][c] == CellState.QUEEN
    }

    for q in given:
        if q in forbidden:
            raise ValueError(f"Given queen on forbidden X cell {q}")
        if conflicts_with_existing(q):
            raise ValueError(f"Conflicting given queens at {q}")
        place(q)  # we ignore returned list; givens are permanent

    # candidate predicate
    def is_candidate(cell: Coord) -> bool:
        r, c = cell
        if row_filled[r]:
            return False
        if c in used_cols:
            return False
        k = region_id[r][c]
        if k in used_regions:
            return False
        if is_blocked(cell):
            return False
        if cell in forbidden:
            return False
        return True

    def candidates_for_row(r: int) -> List[Coord]:
        return [(r, c) for c in range(cols) if is_candidate((r, c))]

    def candidates_for_col(c: int) -> List[Coord]:
        if c in used_cols:
            return []
        return [(r, c) for r in range(rows) if is_candidate((r, c))]

    def candidates_for_region(k: int) -> List[Coord]:
        if k in used_regions:
            return []
        return [cell for cell in region_cells[k] if is_candidate(cell)]

    # propagation (returns list of forced placements with their deltas)
    def propagate() -> Optional[List[Tuple[Coord, List[Coord]]]]:
        forced_stack: List[Tuple[Coord, List[Coord]]] = []

        changed = True
        while changed:
            changed = False

            # rows
            for r in range(rows):
                if row_filled[r]:
                    continue
                cands = candidates_for_row(r)
                if len(cands) == 0:
                    # dead end
                    for cell, nb_delta in reversed(forced_stack):
                        unplace(cell, nb_delta)
                    return None
                if len(cands) == 1:
                    cell = cands[0]
                    nb_delta = place(cell)
                    forced_stack.append((cell, nb_delta))
                    changed = True

            # cols
            for c in range(cols):
                if c in used_cols:
                    continue
                cands = candidates_for_col(c)
                if len(cands) == 0:
                    for cell, nb_delta in reversed(forced_stack):
                        unplace(cell, nb_delta)
                    return None
                if len(cands) == 1:
                    cell = cands[0]
                    nb_delta = place(cell)
                    forced_stack.append((cell, nb_delta))
                    changed = True

            # regions
            for k in all_regions:
                if k in used_regions:
                    continue
                cands = candidates_for_region(k)
                if len(cands) == 0:
                    for cell, nb_delta in reversed(forced_stack):
                        unplace(cell, nb_delta)
                    return None
                if len(cands) == 1:
                    cell = cands[0]
                    nb_delta = place(cell)
                    forced_stack.append((cell, nb_delta))
                    changed = True

        return forced_stack

    def undo_forced(forced_stack: List[Tuple[Coord, List[Coord]]]):
        for cell, nb_delta in reversed(forced_stack):
            unplace(cell, nb_delta)

    # MRV row choice
    def choose_row_mrv() -> int:
        best_r = -1
        best_len = 10**9
        for r in range(rows):
            if row_filled[r]:
                continue
            l = len(candidates_for_row(r))
            if l < best_len:
                best_len = l
                best_r = r
        return best_r

    # LCV ordering by minimal new blocks
    def lcv_order(cands: List[Coord]) -> List[Coord]:
        def impact(cell: Coord) -> int:
            return sum(1 for nb in neighbors[cell] if not is_blocked(nb) and nb not in queens)
        return sorted(cands, key=impact)

    def is_goal() -> bool:
        if len(queens) != rows:
            return False
        # strict final validation: each col and region exactly 1
        col_counts = [0]*cols
        reg_counts: Dict[int, int] = {k: 0 for k in all_regions}
        for r, c in queens:
            col_counts[c] += 1
            reg_counts[region_id[r][c]] += 1
        return all(x == 1 for x in col_counts) and all(v == 1 for v in reg_counts.values())

    def dfs() -> Optional[Set[Coord]]:
        if all(row_filled):
            return queens.copy() if is_goal() else None

        forced = propagate()
        if forced is None:
            return None

        if all(row_filled):
            sol = queens.copy() if is_goal() else None
            undo_forced(forced)
            return sol

        r = choose_row_mrv()
        cands = candidates_for_row(r)
        if not cands:
            undo_forced(forced)
            return None

        for cell in lcv_order(cands):
            nb_delta = place(cell)
            sol = dfs()
            if sol is not None:
                return sol
            unplace(cell, nb_delta)

        undo_forced(forced)
        return None

    sol = dfs()
    if sol is None:
        raise ValueError("No solution found.")
    return sol


if __name__ == "__main__":
    queens_board_selectors = ["#queens-grid", ".queens-grid-no-gap", ".queens-grid"]

    board_html = asyncio.run(
        fetch_rendered_board_html(
            "https://www.linkedin.com/games/queens",
            board_identifier=queens_board_selectors,
            debug=False,
        )
    )

    board = parse_queens_board(board_html)
    print(render_queens_ascii(board, show_regions=False))

    solution = solve_queens(board)
    print("\n=== SOLUTION ===")
    print(render_queens_ascii(board, show_regions=False, solution_queens=solution))

    # also print coords in 1-based for convenience
    print("\nQueen positions (1-based):")
    print(sorted([(r+1, c+1) for (r, c) in solution]))
