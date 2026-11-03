import random
from collections import deque

# ---------- Settings (meter) ----------
NUM_OBSTACLES = 8
OBS_SIZE      = 0.05
OBS_HEIGHT    = 0.20
BOT_WIDTH     = 0.14
MIN_GAP       = 0.25          # only used for the log message now

ARENA_HALF_W  = 0.475
ARENA_HALF_L  = 1.475
X_RANGE       = (-0.38, 0.38)
Y_RANGE       = (-0.85, 0.90)

MIN_DIST      = 0.30          # centre-to-centre (Euclidean) between obstacles
PLAN_RADIUS   = 0.085         # bot half-width (0.07) + margin
GRID          = 0.02


def far_enough(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 >= MIN_DIST ** 2


def _box_dist(px, py, o):
    h = OBS_SIZE / 2.0
    dx = max(abs(px - o[0]) - h, 0.0)
    dy = max(abs(py - o[1]) - h, 0.0)
    return (dx * dx + dy * dy) ** 0.5


def path_exists(obstacles):
    nx = int(2 * ARENA_HALF_W / GRID)
    ny = int(2 * ARENA_HALF_L / GRID)

    def centre(i, j):
        return (-ARENA_HALF_W + (i + 0.5) * GRID, -ARENA_HALF_L + (j + 0.5) * GRID)

    def free(i, j):
        x, y = centre(i, j)
        if abs(x) > ARENA_HALF_W - PLAN_RADIUS:
            return False
        return all(_box_dist(x, y, o) > PLAN_RADIUS for o in obstacles)

    start = (nx // 2, int((-1.20 + ARENA_HALF_L) / GRID))
    if not free(*start):
        return False
    seen = {start}
    q = deque([start])
    goal_j = int((1.10 + ARENA_HALF_L) / GRID)
    while q:
        i, j = q.popleft()
        if j >= goal_j:
            return True
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (i + di, j + dj)
            if 0 <= n[0] < nx and 0 <= n[1] < ny and n not in seen and free(*n):
                seen.add(n)
                q.append(n)
    return False


def pick_positions(rng=random, n=NUM_OBSTACLES):
    for _ in range(2000):
        pts = []
        for _ in range(500):
            p = (rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE))
            if all(far_enough(p, q) for q in pts):
                pts.append(p)
                if len(pts) == n:
                    break
        if len(pts) == n and path_exists(pts):
            return pts
    raise RuntimeError('Could not find a valid obstacle layout')
