"""Navigators for the obstacle course (pure python, no ROS) -> unit-testable.

Pick the algorithm with the LAST line of this file:
    Navigator = RadarNavigator         # servo sweep + 1 TOF (current hardware)
    Navigator = FirstApproach          # scan-and-go (static sonar scan, turn, drive)
    Navigator = Bug2Navigator          # Bug2 + wall following around obstacles
    Navigator = WallFollowNavigator    # simple wall following around obstacles

Note: RadarNavigator needs the servo+TOF hardware (robot.urdf.xacro);
all sonar-based navigators below are kept for reference but the sonars
are gone from the URDF, so RadarNavigator is the active one.


Same interface as the scan-and-go Navigator:
    nav = Bug2Navigator()            # or WallFollowNavigator()
    nav.reset(x0, y0, yaw0)
    v, w = nav.step(fl, fc, fr, l, r, x, y, rel, t)

Goal      : straight ahead along the start heading (the "M-line" = lateral 0).
            Only the color sensor ends the run (the navigator never gives up).
Angles    : relative to the start heading, 0 = straight ahead, + = left.
End wall  : NOT an obstacle (the finish line is in front of it).

States
  GO        drive toward the goal, gently steering back to the M-line
  ALIGN     obstacle hit: turn in place 90 deg away from it (side with more room)
  SIDESTEP  drive sideways a fixed distance (enough to clear a thin pole)
  TURNBACK  turn back to face the goal
  FOLLOW    drive past the obstacle holding heading 0, keeping the side sonar
            a safe distance from it (no circling around thin poles)
  RETURN    (Bug2 only) steer back onto the M-line, then GO
  FINAL     past the last obstacle row -> straight dash

Two variants share all of the above, they differ only in WHEN to leave FOLLOW:
  Bug2Navigator         obstacle is "passed" by odometry (progress beyond the hit
                        point), then RETURN to the M-line before going on (Bug2).
  WallFollowNavigator   obstacle is "passed" when the side sonar saw it and then
                        lost it -> straight back to GO (hand-on-the-wall, Bug0).
"""
import math

INF = 9.9


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class _BugBase:
    # ---- arena geometry (progress is measured from the START position) ----
    WALL_P      = 2.725
    WALL_TOL    = 0.15
    SONAR_X     = 0.10
    FINAL_P     = 2.20
    V_FINAL     = 0.12
    LANE_HALF   = 0.32
    AX_FL, AX_FC, AX_FR = 0.45, 0.0, -0.45
    # ---- hit detection ----
    HIT_FRONT   = 0.15
    HIT_DIAG    = 0.12
    PASS_FRONT  = 0.12    # stricter limits while passing beside an obstacle
    PASS_DIAG   = 0.09
    # ---- GO / RETURN ----
    V_FWD       = 0.15
    K_HEAD      = 2.5
    W_MAX       = 0.8
    K_LAT       = 1.5
    LAT_HEAD_MAX = 0.30
    K_RET       = 2.0
    RET_HEAD_MAX = 0.50
    M_TOL       = 0.04
    # ---- turning in place ----
    K_TURN      = 2.0
    TURN_W_MAX  = 1.0
    TURN_W_MIN  = 0.25
    ALIGN_TOL   = math.radians(3)
    SETTLE      = 0.20
    # ---- sidestep around the obstacle ----
    SIDESTEP    = 0.16    # lateral distance moved (bot half 0.05 + pole 0.025 + margin)
    # ---- passing (FOLLOW) ----
    FOLLOW_SEE  = 0.20    # side sonar sees the obstacle below this
    LOST        = 0.22    # side sonar above this = obstacle no longer beside us
    D_MIN       = 0.065   # push away if the side sonar reads less than this
    K_PUSH      = 10.0
    PASS_PROGRESS = 0.38  # Bug2: axle progress past the hit point = passed
    PASS_MAX    = 0.60    # safety: never follow longer than this
    LOST_LEAVE  = 0.08    # WallFollow: distance driven after losing the obstacle
    NEXT        = 'GO'    # state after FOLLOW (Bug2 overrides)
    T_TURN      = 6.0
    T_SIDE      = 5.0
    T_FOLLOW    = 12.0
    T_RETURN    = 6.0
    DEBUG       = False   # print state + side sonars twice a second

    def __init__(self):
        self.reset(0.0, 0.0, 0.0)

    def reset(self, x0, y0, yaw0):
        self.x0, self.y0, self.yaw0 = x0, y0, yaw0
        self.state = None
        self.obs_side = +1          # +1 = obstacle on the LEFT, -1 = on the RIGHT
        self.hit_p = 0.0
        self.hit_lat = 0.0
        self.hits = 0
        self.seen_side = False
        self.lost_xy = None
        self.t_state = 0.0
        self.ok_since = None
        self.x, self.y, self.rel = x0, y0, 0.0

    @property
    def blocked(self):
        """Compatibility with the scan-and-go Navigator (controller logs it)."""
        return self.hits

    # ---- pose helpers ----
    def progress(self, x, y):
        return (x - self.x0) * math.cos(self.yaw0) + (y - self.y0) * math.sin(self.yaw0)

    def lateral(self, x, y):
        return -(x - self.x0) * math.sin(self.yaw0) + (y - self.y0) * math.cos(self.yaw0)

    def _enter(self, state, t):
        self.state = state
        self.t_state = t
        self.ok_since = None

    def _turn_to(self, target, t):
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

    def _drive_heading(self, target):
        err = wrap(target - self.rel)
        v = self.V_FWD * (1.0 - min(1.0, abs(err) / 0.5))
        return v, clamp(self.K_HEAD * err, -self.W_MAX, self.W_MAX)

    # ---- sensors ----
    def _is_wall(self, d, axis):
        bearing = wrap(self.rel + axis)
        c = math.cos(bearing)
        if c < 0.3:
            return False
        p_s = self.progress(self.x, self.y) + self.SONAR_X * math.cos(self.rel)
        expected = (self.WALL_P - p_s) / c
        return d >= expected - self.WALL_TOL

    def _obst(self, d, axis):
        return INF if self._is_wall(d, axis) else d

    def _side(self, l, r):
        return l if self.obs_side > 0 else r

    # ---- variant hook ----
    def _pass_done(self, prog, gone):
        return False

    # ---- main entry ----
    def step(self, fl, fc, fr, l, r, x, y, rel, t):
        self.x, self.y, self.rel = x, y, rel
        if self.state is None:
            self._enter('GO', t)
        prog = self.progress(x, y)
        lat = self.lateral(x, y)
        if self.state != 'FINAL' and prog >= self.FINAL_P:
            self._enter('FINAL', t)
            print('[nav] final stretch: driving straight to the finish line', flush=True)
        s = self.state
        elapsed = t - self.t_state
        if self.DEBUG and s != 'GO' and int(t * 2) != getattr(self, '_dbg', -1):
            self._dbg = int(t * 2)
            print('[dbg] %s obs_side=%+d fl=%.2f fc=%.2f fr=%.2f l=%.2f r=%.2f rel=%.0fdeg lat=%.2f'
                  % (s, self.obs_side, fl, fc, fr, l, r, math.degrees(rel), lat), flush=True)

        if s == 'FINAL':
            err = wrap(0.0 - rel)
            if fc < 0.06:
                return 0.0, 0.0
            if abs(err) > math.radians(10):
                w = clamp(self.K_TURN * err, -self.TURN_W_MAX, self.TURN_W_MAX)
                if abs(w) < self.TURN_W_MIN:
                    w = math.copysign(self.TURN_W_MIN, w)
                return 0.0, w
            return self.V_FINAL, clamp(self.K_HEAD * err, -self.W_MAX, self.W_MAX)

        # end wall counts as free space
        fl_ = self._obst(fl, self.AX_FL)
        fc_ = self._obst(fc, self.AX_FC)
        fr_ = self._obst(fr, self.AX_FR)

        if s == 'GO' or s == 'RETURN':
            if fc_ < self.HIT_FRONT or fl_ < self.HIT_DIAG or fr_ < self.HIT_DIAG:
                self._on_hit(fl_, fc_, fr_, prog, lat, t)
                return 0.0, 0.0
            if s == 'GO':
                target = clamp(-self.K_LAT * lat, -self.LAT_HEAD_MAX, self.LAT_HEAD_MAX)
            else:
                if abs(lat) < self.M_TOL or elapsed > self.T_RETURN:
                    self._enter('GO', t)
                target = clamp(-self.K_RET * lat, -self.RET_HEAD_MAX, self.RET_HEAD_MAX)
            return self._drive_heading(target)

        if s == 'ALIGN':                     # turn 90 deg away from the obstacle
            w, done = self._turn_to(wrap(-self.obs_side * math.pi / 2), t)
            if done or elapsed > self.T_TURN:
                self._enter('SIDESTEP', t)
                return 0.0, 0.0
            return 0.0, w

        if s == 'SIDESTEP':                  # slide sideways past the pole's edge
            moved = abs(lat - self.hit_lat)
            edge = (-self.obs_side * lat) >= self.LANE_HALF
            if moved >= self.SIDESTEP or edge or fc_ < 0.06 or elapsed > self.T_SIDE:
                self._enter('TURNBACK', t)
                return 0.0, 0.0
            return self._drive_heading(wrap(-self.obs_side * math.pi / 2))

        if s == 'TURNBACK':
            w, done = self._turn_to(0.0, t)
            if done or elapsed > self.T_TURN:
                self.seen_side = False
                self.lost_xy = None
                self._enter('FOLLOW', t)
                return 0.0, 0.0
            return 0.0, w

        if s == 'FOLLOW':                    # pass the obstacle, heading held at 0
            if fc_ < self.PASS_FRONT or fl_ < self.PASS_DIAG or fr_ < self.PASS_DIAG:
                self._on_hit(fl_, fc_, fr_, prog, lat, t)     # another obstacle ahead
                return 0.0, 0.0
            ds = self._side(l, r)
            gone = 0.0
            if ds < self.FOLLOW_SEE:
                self.seen_side = True
                self.lost_xy = None
            elif ds > self.LOST and self.seen_side:
                if self.lost_xy is None:
                    self.lost_xy = (x, y)
                gone = math.hypot(x - self.lost_xy[0], y - self.lost_xy[1])
            if (self._pass_done(prog, gone) or prog >= self.hit_p + self.PASS_MAX
                    or elapsed > self.T_FOLLOW):
                print('[nav] obstacle passed (progress=%.2f lateral=%.2f)' % (prog, lat), flush=True)
                self._enter(self.NEXT, t)
                return 0.0, 0.0
            v, w = self._drive_heading(0.0)
            if ds < self.D_MIN:              # too close to the obstacle: push away
                w += -self.obs_side * self.K_PUSH * (self.D_MIN - ds)
            return v, clamp(w, -self.W_MAX, self.W_MAX)

        return 0.0, 0.0

    def _on_hit(self, fl_, fc_, fr_, prog, lat, t):
        """Obstacle ahead: remember the hit point and pick the side to pass on."""
        self.hit_p, self.hit_lat = prog, lat
        self.hits += 1
        if fr_ < fl_ - 0.03:
            bypass_left = True            # obstacle is more on the right
        elif fl_ < fr_ - 0.03:
            bypass_left = False
        else:
            bypass_left = lat <= 0.0      # more room on the left
        room_left = self.LANE_HALF - lat
        room_right = self.LANE_HALF + lat
        need = self.SIDESTEP - 0.02
        if bypass_left and room_left < need <= room_right:
            bypass_left = False           # not enough room on the preferred side
        elif (not bypass_left) and room_right < need <= room_left:
            bypass_left = True
        self.obs_side = -1 if bypass_left else +1
        print('[nav] hit #%d (progress=%.2f lateral=%.2f): fl=%.2f fc=%.2f fr=%.2f -> pass on the %s'
              % (self.hits, prog, lat, fl_, fc_, fr_, 'left' if bypass_left else 'right'),
              flush=True)
        self._enter('ALIGN', t)


class Bug2Navigator(_BugBase):
    """Bug2: pass the obstacle (odometry says we are beyond it), then steer back
    onto the M-line (RETURN) before going on."""
    NEXT = 'RETURN'

    def _pass_done(self, prog, gone):
        return prog >= self.hit_p + self.PASS_PROGRESS


class WallFollowNavigator(_BugBase):
    """Hand-on-the-wall rule (Bug0 style): pass the obstacle until the side sonar
    saw it and then lost it, then go straight back to GO."""
    NEXT = 'GO'

    def _pass_done(self, prog, gone):
        return self.seen_side and gone >= self.LOST_LEAVE


class FirstApproach:
    """Scan-and-go navigator.

    Cycle  : SCAN (standing still) -> TURN -> DRIVE -> STRAIGHTEN -> SCAN -> ...
    Blocked: SCAN finds nothing safe -> STRAIGHTEN -> BACKUP -> SCAN
    Final  : past the last obstacle row -> FINAL (straight dash; color sensor stops it)
    Angles are relative to the start heading: 0 = straight ahead, + = left.
    The arena end wall is NOT an obstacle (the finish line is in front of it).
    The navigator never gives up: only the color sensor ends the run.
    """
    # ---- arena geometry (progress is measured from the START position) ----
    WALL_P      = 2.725
    WALL_TOL    = 0.15
    WALL_STOP   = 0.12
    SONAR_X     = 0.10
    FINAL_P     = 2.20
    V_FINAL     = 0.12
    # ---- static scan (no rotation) ----
    SCAN_SETTLE = 0.15
    SCAN_DWELL  = 0.35
    SCAN_HALF   = math.radians(40)
    BIN         = math.radians(5)
    OBS_SPREAD   = 0.04
    AXES        = (('fr', -0.45), ('fc', 0.0), ('fl', 0.45))
    # ---- planning ----
    CLEAR_HALF   = 0.09
    STOP_MARGIN  = 0.14
    EDGE_PENALTY = 0.25
    BAD_TOL      = math.radians(12)
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
    # ---- timeouts (s) ----
    T_TURN, T_DRIVE, T_BACK = 6.0, 10.0, 5.0

    def __init__(self):
        self.N = int(round(2 * self.SCAN_HALF / self.BIN)) + 1
        self.reset(0.0, 0.0, 0.0)

    def reset(self, x0, y0, yaw0):
        """x0, y0, yaw0 = odom pose at START (used for progress/lateral)."""
        self.x0, self.y0, self.yaw0 = x0, y0, yaw0
        self.state = None
        self.after = 'SCAN'
        self.blocked = 0
        self.bad = []
        self.pts = []
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
        if self.state != 'FINAL' and self.progress(x, y) >= self.FINAL_P:
            self._enter('FINAL', t)
            print('[nav] final stretch: driving straight to the finish line', flush=True)
        s = self.state
        elapsed = t - self.t_state

        if s == 'FINAL':
            err = wrap(0.0 - rel)
            if fc < 0.06:
                return 0.0, 0.0
            if abs(err) > math.radians(10):
                w = clamp(self.K_TURN * err, -self.TURN_W_MAX, self.TURN_W_MAX)
                if abs(w) < self.TURN_W_MIN:
                    w = math.copysign(self.TURN_W_MIN, w)
                return 0.0, w
            return self.V_FINAL, clamp(self.K_HEAD * err, -self.W_MAX, self.W_MAX)

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
                    self.bad = []
                else:
                    self.bad.append(self.target)
                    print('[nav] drive to %.0f deg cancelled: fc=%.2f fl=%.2f fr=%.2f l=%.2f r=%.2f'
                          % (math.degrees(self.target), fc, fl, fr, l, r), flush=True)
                self.after = 'SCAN'
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
                self.bad = []
                self._enter('SCAN', t)
                return 0.0, 0.0
            return self.V_BACK, clamp(self.K_HEAD * wrap(0.0 - rel), -self.W_MAX, self.W_MAX)

        return 0.0, 0.0

    # ---- static scan + decision ----
    def _scan(self, fl, fc, fr, t):
        el = t - self.t_state
        if el < self.SCAN_SETTLE:
            return 0.0, 0.0
        for name, val in (('fl', fl), ('fc', fc), ('fr', fr)):
            cur = self.acc.get(name)
            self.acc[name] = val if cur is None else min(cur, val)
        if el < self.SCAN_DWELL:
            return 0.0, 0.0
        return self._decide(t)

    def _is_wall(self, d, bearing):
        c = math.cos(bearing)
        if c < 0.3:
            return False
        p_s = self.progress(self.x, self.y) + self.SONAR_X * math.cos(self.rel)
        expected = (self.WALL_P - p_s) / c
        return d >= expected - self.WALL_TOL

    def _build_points(self):
        pts = []
        for name, axis in self.AXES:
            d = self.acc.get(name)
            if d is None:
                continue
            if self._is_wall(d, wrap(self.rel + axis)):
                continue
            for k in (-1, 0, 1):
                pts.append((axis + k * self.OBS_SPREAD, d))
        return pts

    def _free_length(self, th):
        clear = self._clear()
        length = self.DRIVE_MAX
        for ang, d in self.pts:
            dth = ang - th
            along = d * math.cos(dth)
            side = abs(d * math.sin(dth))
            if along > 0.0 and side < clear:
                length = min(length, along - self.STOP_MARGIN)
        return length

    def _decide(self, t):
        self.pts = self._build_points()
        lat = self.lateral(self.x, self.y)
        p_s = self.progress(self.x, self.y) + self.SONAR_X * math.cos(self.rel)
        best, best_score, best_len = None, 0.0, 0.0
        for i in range(self.N):
            th = self._ang(i)
            a = wrap(self.rel + th)
            if any(abs(wrap(a - b)) < self.BAD_TOL for b in self.bad):
                continue
            ca, sa = math.cos(a), math.sin(a)
            length = self._free_length(th)
            if ca > 0.2:
                length = min(length, (self.WALL_P - self.WALL_STOP - p_s) / ca)
            if sa > 1e-3:
                length = min(length, (self.LANE_HALF - lat) / sa)
            elif sa < -1e-3:
                length = min(length, (-self.LANE_HALF - lat) / sa)
            gain = length * ca
            if gain < self.MIN_PROGRESS:
                continue
            score = gain - self.EDGE_PENALTY * abs(a)
            if best is None or score > best_score:
                best, best_score, best_len = a, score, length

        if best is None:
            self.blocked += 1
            self.after = 'BACKUP'
            print('[nav] blocked (progress=%.2f lateral=%.2f): fl=%.2f fc=%.2f fr=%.2f'
                  % (self.progress(self.x, self.y), lat, self.acc.get('fl', 0.0),
                     self.acc.get('fc', 0.0), self.acc.get('fr', 0.0)), flush=True)
            self._enter('STRAIGHTEN', t)
            return 0.0, 0.0
        self.target, self.length = best, best_len
        self._enter('TURN', t)
        return 0.0, 0.0


class RadarNavigator:
    """Servo-scanning navigator: one TOF ranging on a sweeping servo head.

    The head sweeps left-right continuously (commanded servo angle =
    SWEEP_AMP * sin(2*pi*t/T), a pure function of time since START). Every
    range reading is tagged with the servo angle it was taken at, so
    obstacles are located by ANGLE precisely (~ +-BIN/2) - the robot never
    has to rotate to scan, and there is no sonar-cone fuzz.

    Cycle ("go to the farthest gap, cross the obstacle, straighten up"):
      SCAN        stand still; the servo sweep fills a polar range map
      (DECIDE)    pick the candidate heading with the longest free run
                  ("farthest"), planned with a margin so the leg CROSSES
                  the obstacle
      TURN        rotate the robot to that heading
      DRIVE       drive the planned length, servo keeps sweeping; a fresh
                  close reading inside the planned corridor cancels -> rescan
      STRAIGHTEN  turn back to the goal heading (0)   ("make itself straight")
      BACKUP      stuck (blocked): reverse a little, then rescan
      FINAL       past the last obstacle row -> straight dash; the camera
                  seeing the green finish line ends the run

    Angles are relative to the start heading: 0 = straight ahead, + = left.
    The arena end wall is NOT an obstacle (the finish line is in front of it).
    """

    # ---- arena geometry (progress measured from the START position) ----
    WALL_P      = 2.725
    WALL_TOL    = 0.15
    WALL_STOP   = 0.12
    BEAM_X      = 0.10    # beam pivot ahead of the axle (servo joint x)
    FINAL_P     = 2.20
    V_FINAL     = 0.12
    # ---- servo sweep ----
    SWEEP_AMP   = 1.05    # rad, left/right of straight ahead (joint limit 1.4)
    SWEEP_T     = 3.6     # s, full sweep period (slow so the head tracks it)
    MAX_RANGE   = 3.0
    BIN         = 0.10    # angular resolution of the sweep map (rad)
    SCAN_SETTLE = 0.15    # s standing still before the sweep is accepted
    SCAN_DWELL  = 4.00    # s scan time (one full sweep + margin); then decide
    # ---- planning ----
    CLEAR_HALF   = 0.10   # corridor half-width the bot needs
    STOP_MARGIN  = 0.14   # beam-to-obstacle distance where a leg ends
    EDGE_PENALTY = 0.25   # large headings (sideways) cost more
    BAD_TOL      = math.radians(12)   # a failed heading bans +/-12 deg
    DRIVE_MAX    = 0.60
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
    # ---- mid-drive safety veto (fresh sweep readings while driving) ----
    VETO_HALF    = 0.28   # a reading within this angle of the heading ...
    VETO_D       = 0.11   # ... closer than this cancels the leg -> rescan
    VETO_T       = 0.40   # readings older than this are stale
    # ---- backup ----
    V_BACK       = -0.08
    BACK_STEPS   = (0.10, 0.15, 0.20, 0.25)
    # ---- timeouts (s) ----
    T_TURN, T_DRIVE, T_BACK = 6.0, 10.0, 5.0
    DEBUG        = False  # print map + decision each cycle

    def __init__(self):
        self.reset(0.0, 0.0, 0.0)

    def reset(self, x0, y0, yaw0):
        """x0, y0, yaw0 = odom pose at START (used for progress/lateral)."""
        self.x0, self.y0, self.yaw0 = x0, y0, yaw0
        self.state = None
        self.after = 'SCAN'
        self.blocked = 0
        self.bad = []
        self.acc = {}          # bin index -> shortest range seen this scan
        self.recent = []       # fresh (t, angle-vs-heading, range) for the veto
        self.pts = []
        self.target = 0.0
        self.length = 0.0
        self.t_state = 0.0
        self.ok_since = None
        self.sx = self.sy = 0.0
        self.x, self.y, self.rel = x0, y0, 0.0

    # ---- servo command (pure function of t since START) ----
    def servo_cmd(self, t):
        if self.state == 'FINAL':
            return 0.0                       # head straight ahead for the dash
        return self.SWEEP_AMP * math.sin(2.0 * math.pi * t / self.SWEEP_T)

    # ---- one TOF reading (called by the controller per range msg) ----
    def tof(self, a, d, t):
        """a = servo angle vs the HEADING (rad, + left), d = range (m), t = s."""
        if not math.isfinite(d) or d <= 0.0:
            return
        d = min(d, self.MAX_RANGE)
        b = int(round(a / self.BIN))
        cur = self.acc.get(b)
        self.acc[b] = min(cur, d) if cur is not None else d
        self.recent.append((t, a, d))
        cutoff = t - self.VETO_T
        while self.recent and self.recent[0][0] < cutoff:
            self.recent.pop(0)

    def _center(self, t):
        """Closest reading inside ~+-0.2 rad among the last 0.3 s of samples."""
        best = self.MAX_RANGE
        for ts, a, d in self.recent:
            if t - ts > 0.30:
                continue
            if abs(a) < 0.2 and d < best:
                best = d
        return best

    # ---- pose helpers (odom frame) ----
    def progress(self, x, y):
        return (x - self.x0) * math.cos(self.yaw0) + (y - self.y0) * math.sin(self.yaw0)

    def lateral(self, x, y):
        return -(x - self.x0) * math.sin(self.yaw0) + (y - self.y0) * math.cos(self.yaw0)

    def _enter(self, state, t):
        self.state = state
        self.t_state = t
        self.ok_since = None
        self.sx, self.sy = self.x, self.y
        if state == 'SCAN':
            self.acc = {}

    def _turn_to(self, target, t):
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

    # ---- wall filter (the end wall is not an obstacle) ----
    def _is_wall(self, d, bearing):
        c = math.cos(bearing)
        if c < 0.3:
            return False
        p_b = self.progress(self.x, self.y) + self.BEAM_X * math.cos(self.rel)
        expected = (self.WALL_P - p_b) / c
        return d >= expected - self.WALL_TOL

    def _build_points(self):
        pts = []
        for b, d in sorted(self.acc.items()):
            a = b * self.BIN
            if self._is_wall(d, wrap(self.rel + a)):
                continue
            pts.append((a, d))
        return pts

    def _free_length(self, th):
        clear = self.CLEAR_HALF
        length = self.DRIVE_MAX
        for a, d in self.pts:
            dth = a - th
            along = d * math.cos(dth)
            side = abs(d * math.sin(dth))
            if along > 0.0 and side < clear:
                length = min(length, along - self.STOP_MARGIN)
        return length

    def _veto(self, t):
        """True if a fresh front-sector reading is dangerously close."""
        for ts, a, d in self.recent:
            if t - ts > self.VETO_T:
                continue
            if abs(a) <= self.VETO_HALF and d < self.VETO_D:
                return True
        return False

    # ---- main entry ----
    def step(self, fl=None, fc=None, fr=None, l=None, r=None, x=None, y=None,
             rel=None, t=None):
        self.x, self.y, self.rel = x, y, rel
        if self.state is None:
            self._enter('SCAN', t)
        if self.state != 'FINAL' and self.progress(x, y) >= self.FINAL_P:
            self._enter('FINAL', t)
            print('[nav] final stretch: driving straight to the finish line', flush=True)
        s = self.state
        elapsed = t - self.t_state

        if s == 'FINAL':
            err = wrap(0.0 - rel)
            if self._center(t) < 0.06:
                return 0.0, 0.0
            if abs(err) > math.radians(10):
                w = clamp(self.K_TURN * err, -self.TURN_W_MAX, self.TURN_W_MAX)
                if abs(w) < self.TURN_W_MIN:
                    w = math.copysign(self.TURN_W_MIN, w)
                return 0.0, w
            return self.V_FINAL, clamp(self.K_HEAD * err, -self.W_MAX, self.W_MAX)

        if s == 'SCAN':
            if elapsed < self.SCAN_SETTLE:
                return 0.0, 0.0
            if elapsed < self.SCAN_DWELL:
                return 0.0, 0.0
            return self._decide(t)

        if s == 'TURN':
            w, done = self._turn_to(self.target, t)
            if done or elapsed > self.T_TURN:
                self._enter('DRIVE', t)
                return 0.0, 0.0
            return 0.0, w

        if s == 'DRIVE':
            moved = math.hypot(x - self.sx, y - self.sy)
            if self._veto(t):
                if moved >= 0.05:
                    self.blocked = 0
                    self.bad = []
                else:
                    self.bad.append(self.target)
                    print('[nav] drive to %.0f deg cancelled: too close ahead'
                          % math.degrees(self.target), flush=True)
                self.after = 'SCAN'
                self._enter('STRAIGHTEN', t)
                return 0.0, 0.0
            if moved >= self.length or elapsed > self.T_DRIVE:
                if moved >= 0.05:
                    self.blocked = 0
                    self.bad = []
                else:
                    self.bad.append(self.target)
                    print('[nav] drive to %.0f deg stalled at %.2f m'
                          % (math.degrees(self.target), moved), flush=True)
                self.after = 'SCAN'
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
                self.bad = []
                self._enter('SCAN', t)
                return 0.0, 0.0
            return self.V_BACK, clamp(self.K_HEAD * wrap(0.0 - rel), -self.W_MAX, self.W_MAX)

        return 0.0, 0.0

    # ---- polar sweep map -> choose the farthest usable heading ----
    def _decide(self, t):
        self.pts = self._build_points()
        lat = self.lateral(self.x, self.y)
        p_b = self.progress(self.x, self.y) + self.BEAM_X * math.cos(self.rel)
        best, best_score, best_len = None, 0.0, 0.0
        n_cand = int(round(self.SWEEP_AMP / 0.1))   # candidates every 0.1 rad
        for i in range(-n_cand, n_cand + 1):
            th = i * 0.1                            # heading-frame candidate
            a = wrap(self.rel + th)                 # same heading, start-frame
            if any(abs(wrap(a - bb)) < self.BAD_TOL for bb in self.bad):
                continue
            ca, sa = math.cos(a), math.sin(a)
            length = self._free_length(th)
            if ca > 0.2:
                length = min(length, (self.WALL_P - self.WALL_STOP - p_b) / ca)
            if sa > 1e-3:
                length = min(length, (self.LANE_HALF - lat) / sa)
            elif sa < -1e-3:
                length = min(length, (-self.LANE_HALF - lat) / sa)
            gain = length * ca
            if gain < self.MIN_PROGRESS:
                continue
            score = gain - self.EDGE_PENALTY * abs(a)
            if best is None or score > best_score:
                best, best_score, best_len = a, score, length

        if best is None:
            self.blocked += 1
            self.after = 'BACKUP'
            print('[nav] blocked (progress=%.2f lateral=%.2f, %d bins seen)'
                  % (self.progress(self.x, self.y), lat, len(self.acc)), flush=True)
            self._enter('STRAIGHTEN', t)
            return 0.0, 0.0
        self.target, self.length = best, best_len
        if self.DEBUG:
            print('[nav] map %s' % {i * self.BIN: round(d, 2)
                                    for i, d in sorted(self.acc.items())}, flush=True)
        print('[nav] farthest gap: heading %.0f deg, %.2f m free'
              % (math.degrees(best), best_len), flush=True)
        self._enter('TURN', t)
        return 0.0, 0.0


# ======================================================================
# CHOOSE THE ALGORITHM HERE (change only this line, then colcon build):
#   RadarNavigator       -> servo sweep + 1 TOF (CURRENT HARDWARE)
#   FirstApproach        -> scan-and-go (needs the old sonars - reference only)
#   Bug2Navigator        -> Bug2 + wall following (needs the old sonars)
#   WallFollowNavigator  -> wall following (needs the old sonars)
# ======================================================================
Navigator = RadarNavigator
