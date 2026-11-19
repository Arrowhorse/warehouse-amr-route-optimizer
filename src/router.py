import heapq
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

Coord = Tuple[int, int]


@dataclass
class RouteResult:
    algorithm: str
    path: List[Coord]
    actions: List[str]
    total_steps: int
    nodes_expanded: int
    compute_ms: float
    cache_hits: int = 0


class WarehouseFloor:
    """Represents a 2D warehouse grid with shelving obstacles and order pick locations."""

    MOVES = {
        "UP": (0, 1),
        "DOWN": (0, -1),
        "RIGHT": (1, 0),
        "LEFT": (-1, 0),
    }

    def __init__(self, width: int, height: int, walls: Set[Coord], start: Coord, orders: Set[Coord]):
        self.width = width
        self.height = height
        self.walls = walls
        self.start = start
        self.orders = frozenset(orders)

    def in_bounds(self, pos: Coord) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, pos: Coord) -> bool:
        return pos not in self.walls

    def neighbors(self, pos: Coord) -> List[Tuple[Coord, str]]:
        x, y = pos
        result = []
        for action, (dx, dy) in self.MOVES.items():
            nxt = (x + dx, y + dy)
            if self.in_bounds(nxt) and self.passable(nxt):
                result.append((nxt, action))
        return result


class WarehouseRouter:
    """Multi-stop warehouse order-picking router comparing Optimal A* (MST heuristic) vs. Greedy BFS."""

    def __init__(self, floor: WarehouseFloor):
        self.floor = floor
        self.md_cache: Dict[Tuple[Coord, Coord], int] = {}
        self.mst_cache: Dict[FrozenSet[Coord], int] = {}
        self.cache_hits = 0

    def _manhattan(self, a: Coord, b: Coord) -> int:
        key = (a, b) if a <= b else (b, a)
        if key not in self.md_cache:
            self.md_cache[key] = abs(a[0] - b[0]) + abs(a[1] - b[1])
        return self.md_cache[key]

    def _mst_heuristic(self, pos: Coord, remaining: FrozenSet[Coord]) -> int:
        """
        Admissible lower bound ported from FoodSearchProblem:
        h(state) = min_dist(robot, remaining_items) + Prim's MST weight over remaining_items.
        """
        if not remaining:
            return 0

        if remaining in self.mst_cache:
            self.cache_hits += 1
            mst_weight = self.mst_cache[remaining]
        else:
            items = list(remaining)
            if len(items) <= 1:
                mst_weight = 0
            else:
                tree = {items[0]}
                unvisited = set(items[1:])
                mst_weight = 0
                while unvisited:
                    best_d = None
                    best_v = None
                    for v in unvisited:
                        d = min(self._manhattan(v, u) for u in tree)
                        if best_d is None or d < best_d:
                            best_d, best_v = d, v
                    mst_weight += best_d
                    tree.add(best_v)
                    unvisited.remove(best_v)
            self.mst_cache[remaining] = mst_weight

        entry_dist = min(self._manhattan(pos, item) for item in remaining)
        return entry_dist + mst_weight

    def solve_astar_mst(self, max_expansions: int = 150_000) -> Optional[RouteResult]:
        """Finds globally shortest pick tour using A* with the cached Prim's MST heuristic."""
        t0 = time.perf_counter()
        self.cache_hits = 0
        initial_remaining = self.floor.orders - {self.floor.start}
        start_state = (self.floor.start, initial_remaining)

        frontier = []
        h0 = self._mst_heuristic(self.floor.start, initial_remaining)
        heapq.heappush(frontier, (h0, 0, start_state))

        came_from: Dict[Tuple[Coord, FrozenSet[Coord]], Tuple[Optional[Tuple[Coord, FrozenSet[Coord]]], Optional[str]]] = {
            start_state: (None, None)
        }
        cost_so_far = {start_state: 0}
        nodes_expanded = 0

        while frontier:
            _, g, current_state = heapq.heappop(frontier)
            pos, remaining = current_state

            if not remaining:
                path, actions = self._reconstruct_multi_path(came_from, current_state)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                return RouteResult(
                    algorithm="Optimal A* (Prim's MST Heuristic)",
                    path=path,
                    actions=actions,
                    total_steps=len(actions),
                    nodes_expanded=nodes_expanded,
                    compute_ms=round(elapsed_ms, 2),
                    cache_hits=self.cache_hits,
                )

            if g > cost_so_far.get(current_state, float("inf")):
                continue

            nodes_expanded += 1
            if nodes_expanded > max_expansions:
                return None

            for nxt_pos, action in self.floor.neighbors(pos):
                nxt_remaining = remaining - {nxt_pos} if nxt_pos in remaining else remaining
                nxt_state = (nxt_pos, nxt_remaining)
                new_g = g + 1

                if new_g < cost_so_far.get(nxt_state, float("inf")):
                    cost_so_far[nxt_state] = new_g
                    priority = new_g + self._mst_heuristic(nxt_pos, nxt_remaining)
                    came_from[nxt_state] = (current_state, action)
                    heapq.heappush(frontier, (priority, new_g, nxt_state))

        return None

    def solve_greedy_bfs(self) -> Optional[RouteResult]:
        """Baseline ported from ClosestDotSearchAgent: repeatedly runs BFS to the nearest uncollected item."""
        t0 = time.perf_counter()
        curr_pos = self.floor.start
        remaining = set(self.floor.orders) - {curr_pos}
        full_path = [curr_pos]
        full_actions: List[str] = []
        total_expanded = 0

        while remaining:
            queue = deque([curr_pos])
            parent: Dict[Coord, Tuple[Optional[Coord], Optional[str]]] = {curr_pos: (None, None)}
            found_target = None

            while queue:
                pos = queue.popleft()
                total_expanded += 1
                if pos in remaining:
                    found_target = pos
                    break
                for nxt_pos, action in self.floor.neighbors(pos):
                    if nxt_pos not in parent:
                        parent[nxt_pos] = (pos, action)
                        queue.append(nxt_pos)

            if found_target is None:
                return None

            # Reconstruct segment
            seg_path = []
            seg_actions = []
            cur = found_target
            while cur != curr_pos:
                prev, act = parent[cur]
                seg_path.append(cur)
                seg_actions.append(act)
                cur = prev
            seg_path.reverse()
            seg_actions.reverse()

            full_path.extend(seg_path)
            full_actions.extend(seg_actions)
            remaining.remove(found_target)
            curr_pos = found_target

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return RouteResult(
            algorithm="Greedy Nearest-Item (Repeated BFS)",
            path=full_path,
            actions=full_actions,
            total_steps=len(full_actions),
            nodes_expanded=total_expanded,
            compute_ms=round(elapsed_ms, 2),
        )

    @staticmethod
    def _reconstruct_multi_path(came_from, end_state):
        path = []
        actions = []
        cur = end_state
        while cur is not None:
            path.append(cur[0])
            prev, act = came_from[cur]
            if act is not None:
                actions.append(act)
            cur = prev
        path.reverse()
        actions.reverse()
        return path, actions