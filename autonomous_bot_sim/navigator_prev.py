"""Scan-and-go navigator (pure python, no ROS) -> unit-testable.

Cycle  : SCAN (standing still) -> TURN -> DRIVE -> STRAIGHTEN -> SCAN -> ...
Blocked: SCAN finds nothing safe -> STRAIGHTEN -> BACKUP -> SCAN
Angles are relative to the start heading: 0 = straight ahead, + = left.
The arena end wall is NOT an obstacle (the finish line is in front of it).
"""
import math


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class Navigator:
    # ---- arena geometry (progress is measured from the START position) ----
    WALL_P      = 2.725    # end wall inner face: 1.25 (start line) + 1.475
    WALL_TOL    = 0.15    # reading within this of the expected wall distance = "wall"
    WALL_STOP   = 0.12    # never plan closer than this (sonar to end wall)
    SONAR_X     = 0.10    # fc sonar ahead of the axle (urdf sonar_fc x; 0.15 if not shrunk)
    # ---- static scan (no rotation) ----
    SCAN_SETTLE = 0.15    # s standing still before sampling
    SCAN_DWELL  = 0.35    # s total scan time
    SCAN_HALF   = math.radians(40)
    BIN         = math.radians(5)
    CONE        = 0.26    # sonar half cone (rad), same as the URDF
    AXES        = (('fr', -0.45), ('fc', 0.0), ('fl', 0.45))    # sonar yaw (rad)
    # ---- planning ----
    CLEAR_HALF   = 0.11    # corridor half-width the bot needs
    STOP_MARGIN  = 0.15    # sonar-to-obstacle distance where a segment ends
    DRIVE_MAX    = 0.70
    MIN_PROGRESS = 0.08
    LANE_HALF    = 0.32
    # ---- turning ----
    K_TURN       = 2.0
    TURN_W_MAX   = 1.0
    TURN_W_MIN   = 0.25
    ALIGN_TOL    = math.radians(3)
    SETTLE       = 0.25
    # ---- driving ----
    V_FWD        = 0.15
    K_HEAD       = 2.5
    W_MAX        = 0.8
    STOP_FRONT   = 0.13
    STOP_DIAG    = 0.11
    STOP_SIDE    = 0.03
    # ---- backup ----
    V_BACK       = -0.08
    BACK_STEPS   = (0.10, 0.15, 0.20, 0.25)
    MAX_BLOCKED  = 6
    # ---- timeouts (s) ----
    T_TURN, T_DRIVE, T_BACK = 6.0, 10.0, 5.0

    def __init__(self):
        self.N = int(round(2 * self.SCAN_HALF / self.BIN)) + 1
        self.reset(0.0, 0.0, 0.0)

    def reset(self, x0, y0, yaw0):
        """x0, y0, yaw0 = odom pose at START (used for progress/lateral)."""
        self.x0, self.y0, self.yaw0 = x0, y0, yaw0
        self.state = None             # first step() enters SCAN
        self.after = 'SCAN'            # what STRAIGHTEN leads to
        self.blocked = 0
        self.bins = [None] * self.N
        self.acc = {}
        self.t_state = 0.0
        self.ok_since = None
        self.sx = self.sy = 0.0
        self.target = 0.0
        self.length = 0.0
        self.x, self.y, self.rel = x0, y0, 0.0

    # ---- pose helpers (odom frame) ----
    def progress(self, x, y):
        return (x - self.x0) * math.cos(self.yaw0) + (y - self.y0) * math.sin(self.yaw0)

    def lateral(self, x, y):
        return -(x - self.x0) * math.sin(self.yaw0) + (y - self.y0) * math.cos(self.yaw0)

    def _ang(self, i):
        return -self.SCAN_HALF + i * self.BIN

    def _clear(self):
        """Corridor half-width. Shrinks a little after repeated blocks so the
        bot can squeeze through instead of repeating the same failure (min 0.09)."""
        return max(0.09, self.CLEAR_HALF - 0.01 * min(self.blocked, 2))

    def _enter(self, state, t):
        self.state = state
        self.t_state = t
        self.ok_since = None
        self.sx, self.sy = self.x, self.y
        if state == 'SCAN':
            self.acc = {}
        elif state == 'BACKUP':
            k = min(max(self.blocked, 1), len(self.BACK_STEPS)) - 1
            self.length = self.BACK_STEPS[k]

    def _turn_to(self, target, t):
        """Turn in place to a heading. Returns (w, done)."""
        err = wrap(target - self.rel)
        if abs(err) < self.ALIGN_TOL:
            if self.ok_since is None:
                self.ok_since = t
            return 0.0, (t - self.ok_since) >= self.SETTLE
        self.ok_since = None
        w = clamp(self.K_TURN * err, -self.TURN_W_MAX, self.TURN_W_MAX)
        if abs(w) < self.TURN_W_MIN:
            w = math.copysign(self.TURN_W_MIN, w)
        return w, False

    # ---- main entry ----
    def step(self, fl, fc, fr, l, r, x, y, rel, t):
        """rel = heading relative to start (from IMU). t = seconds. Returns (v, w)."""
        self.x, self.y, self.rel = x, y, rel
        if self.state is None:
            self._enter('SCAN', t)
        s = self.state
        elapsed = t - self.t_state

        if s == 'SCAN':
            return self._scan(fl, fc, fr, t)

        if s == 'TURN':
            w, done = self._turn_to(self.target, t)
            if done or elapsed > self.T_TURN:
                self._enter('DRIVE', t)
                return 0.0, 0.0
            return 0.0, w

        if s == 'DRIVE':
            moved = math.hypot(x - self.sx, y - self.sy)
            danger = (fc < self.STOP_FRONT or fl < self.STOP_DIAG or fr < self.STOP_DIAG
                      or l < self.STOP_SIDE or r < self.STOP_SIDE)
            if moved >= self.length or danger or elapsed > self.T_DRIVE:
                if moved >= 0.05:
                    self.blocked = 0
                    self.after = 'SCAN'
                else:                       # could not move at all
                    self.blocked += 1
                    self.after = 'BACKUP'
                self._enter('STRAIGHTEN', t)
                return 0.0, 0.0
            err = wrap(self.target - rel)
            v = self.V_FWD * (1.0 - min(1.0, abs(err) / 0.5))
            return v, clamp(self.K_HEAD * err, -self.W_MAX, self.W_MAX)

        if s == 'STRAIGHTEN':
            w, done = self._turn_to(0.0, t)
            if done or elapsed > self.T_TURN:
                self._enter(self.after, t)
                return 0.0, 0.0
            return 0.0, w

        if s == 'BACKUP':
            moved = math.hypot(x - self.sx, y - self.sy)
            if moved >= self.length or elapsed > self.T_BACK:
                self._enter('SCAN', t)
                return 0.0, 0.0
            return self.V_BACK, clamp(self.K_HEAD * wrap(0.0 - rel), -self.W_MAX, self.W_MAX)

        return 0.0, 0.0                    # fallback

    # ---- static scan + decision ----
    def _scan(self, fl, fc, fr, t):
        el = t - self.t_state
        if el < self.SCAN_SETTLE:
            return 0.0, 0.0                    # let the sonars settle
        for name, val in (('fl', fl), ('fc', fc), ('fr', fr)):
            cur = self.acc.get(name)
            self.acc[name] = val if cur is None else min(cur, val)    # conservative
        if el < self.SCAN_DWELL:
            return 0.0, 0.0
        return self._decide(t)

    def _is_wall(self, d, bearing):
        """True if a reading along this bearing (rel. to lane axis) is just the end wall."""
        c = math.cos(bearing)
        if c < 0.3:
            return False
        p_s = self.progress(self.x, self.y) + self.SONAR_X * math.cos(self.rel)
        expected = (self.WALL_P - p_s) / c
        return d >= expected - self.WALL_TOL

    def _build_bins(self):
        """Fill the distance profile from the 3 front sonars (each covers a cone)."""
        bins = [None] * self.N
        for name, axis in self.AXES:
            d = self.acc.get(name)
            if d is None:
                continue
            if self._is_wall(d, wrap(self.rel + axis)):
                d = 2.0                        # end wall = free space
            for i in range(self.N):
                if abs(self._ang(i) - axis) <= self.CONE:
                    bins[i] = d if bins[i] is None else min(bins[i], d)
        return bins

    def _free_length(self, i):
        """How far the bot can drive along bin i's heading with its full width."""
        clear = self._clear()
        th = self._ang(i)
        length = self.DRIVE_MAX
        for j, d in enumerate(self.bins):
            if d is None:
                continue
            dth = self._ang(j) - th
            along = d * math.cos(dth)
            side = abs(d * math.sin(dth))
            if along > 0.0 and side < clear:
                length = min(length, along - self.STOP_MARGIN)
        return length

    def _decide(self, t):
        self.bins = self._build_bins()
        lat = self.lateral(self.x, self.y)
        p_s = self.progress(self.x, self.y) + self.SONAR_X * math.cos(self.rel)
        best, best_score, best_len = None, 0.0, 0.0
        for i in range(self.N):
            if self.bins[i] is None:
                continue
            a = wrap(self.rel + self._ang(i))       # heading relative to the lane axis
            ca, sa = math.cos(a), math.sin(a)
            length = self._free_length(i)
            if ca > 0.2:                            # never plan into the end wall
                length = min(length, (self.WALL_P - self.WALL_STOP - p_s) / ca)
            if sa > 1e-3:                           # stay inside the lane
                length = min(length, (self.LANE_HALF - lat) / sa)
            elif sa < -1e-3:
                length = min(length, (-self.LANE_HALF - lat) / sa)
            gain = length * ca                      # forward progress of this segment
            if gain < self.MIN_PROGRESS:
                continue
            score = gain - 0.05 * abs(a)            # tie -> prefer straighter
            if best is None or score > best_score:
                best, best_score, best_len = a, score, length

        if best is None:                            # completely covered -> back up
            self.blocked += 1
            self.after = 'BACKUP'
            self._enter('STRAIGHTEN', t)
            return 0.0, 0.0
        self.target, self.length = best, best_len
        self._enter('TURN', t)
        return 0.0, 0.0
