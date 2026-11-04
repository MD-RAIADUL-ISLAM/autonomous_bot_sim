#!/usr/bin/env python3
"""ROS 2 wrapper around RadarNavigator.

Inputs : /tof_range (1 TOF on a sweeping servo head), /odom, /imu/data,
         /front_camera/image_raw (tilted top camera)
Outputs: /cmd_vel, /set_joint_trajectory (servo head position commands)

How the sweep works: the navigator provides the commanded servo angle as a
pure function of time since START; the controller publishes it to the servo
joint and remembers the (publish time, angle) pairs.  Every TOF reading is
tagged with the angle of the LAST command published before its message
stamp - in Gazebo the joint holds exactly that last commanded position, so
this tag is the angle the beam actually pointed at, and obstacle locations
are known precisely.

Finish : the tilted camera sees the green finish line on the floor.
         Only the BOTTOM rows of the image look at the floor just in front
         of the robot (~0.19-0.21 m), so a run only ends when the strip is
         at the robot's front edge, not metres away.
"""
import math
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Range, Image, Imu, JointState
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from autonomous_bot_sim.navigator import RadarNavigator, wrap

# ---------- settings ----------
SERVO_JOINT = 'servo_joint'
LOOP_PERIOD = 0.05            # 20 Hz

USE_ODOM_FALLBACK = False     # True = also stop when odometry says the line was passed

# Green finish-line detector (camera tilted 0.32 rad down at 0.185 m height):
# only the bottom rows of the 320x240 image - they watch the floor ~0.19 to
# 0.21 m ahead. The 4 cm strip fills that window when it is at the robot's
# front edge, so this fires with the bumper on the line, not metres early.
GREEN_CAM_TILT = 0.32         # rad, must match cam_tilt in the URDF
GREEN_ROWS     = 14           # bottom rows that look at the near floor
GREEN_ROW_MIN  = 14           # green rows needed inside that window
GREEN_FRAMES   = 2            # consecutive frames (noise filter)

# Gazebo sensor/odom publishers use BEST_EFFORT. Subscribers must match.
SENSOR_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST, depth=5)
CMD_QOS = QoSProfile(depth=10)


class AutonomousController(Node):
    def __init__(self):
        super().__init__('autonomous_controller')
        self.nav = RadarNavigator()

        self.pub = self.create_publisher(Twist, '/cmd_vel', CMD_QOS)
        self.path_pub = self.create_publisher(Path, '/trail', 1)
        self.servo_pub = self.create_publisher(JointTrajectory,
                                               '/set_joint_trajectory', CMD_QOS)

        self.create_subscription(Odometry, '/odom', self.odom_cb, SENSOR_QOS)
        self.create_subscription(Bool, '/start_button', self.start_cb, CMD_QOS)

        self.green_frames = 0
        self.create_subscription(Image, '/front_camera/image_raw',
                                 self.color_cb, SENSOR_QOS)

        self.have_tof = False
        self.create_subscription(Range, '/tof_range', self.tof_cb, SENSOR_QOS)

        self.have_odom = False
        self.x = self.y = self.yaw = 0.0

        self.have_imu = False
        self.imu_yaw = self.imu_yaw0 = 0.0
        self.create_subscription(Imu, '/imu/data', self.imu_cb, SENSOR_QOS)

        # ACTUAL servo angle over time, from /joint_states. The beam angle of
        # every TOF reading is taken from here (interpolated to the reading's
        # stamp) - this is exact, no matter how the servo plugin paces itself.
        self.servo_hist = deque(maxlen=256)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, SENSOR_QOS)

        self.state = 'IDLE'       # IDLE -> RUNNING -> FINISHED
        self.start_time = None
        self.path = Path()
        self.path.header.frame_id = 'odom'
        self.last_path_t = 0.0

        self.create_timer(LOOP_PERIOD, self.control_loop)
        self.get_logger().info(
            'Controller ready (servo+TOF). Waiting for START button...')

    # ---------- callbacks ----------
    def joint_cb(self, msg):
        """Record the servo head's ACTUAL angle over time (30 Hz)."""
        if SERVO_JOINT not in msg.name:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.servo_hist.append((t, msg.position[msg.name.index(SERVO_JOINT)]))

    def beam_angle_at(self, t_msg):
        """Servo angle (rad vs heading) at time t_msg: linear interpolation
        between the two nearest /joint_states samples. None if no data yet."""
        hist = self.servo_hist
        if not hist:
            return None
        # find the last sample with t <= t_msg
        i = len(hist) - 1
        while i > 0 and hist[i][0] > t_msg:
            i -= 1
        t1, a1 = hist[i]
        if t1 > t_msg:                        # reading older than all samples
            return None
        if i == len(hist) - 1:
            return a1                         # no later sample to interpolate
        t2, a2 = hist[i + 1]
        if t2 <= t1:
            return a1
        f = (t_msg - t1) / (t2 - t1)
        return a1 + f * (a2 - a1)

    def tof_cb(self, msg):
        r = float(msg.range)
        if not math.isfinite(r) or r > msg.max_range:
            r = float(msg.max_range)          # nothing detected
        if r <= 0.0:
            return
        t_msg = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        a = self.beam_angle_at(t_msg)
        if a is None:                         # reading older than our samples
            return
        self.nav.tof(a, r, t_msg)
        self.have_tof = True

    def odom_cb(self, msg):
        p = msg.pose.pose
        self.x, self.y = float(p.position.x), float(p.position.y)
        q = p.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.have_odom = True

    def imu_cb(self, msg):
        q = msg.orientation
        self.imu_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                  1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.have_imu = True

    def heading(self):
        """Heading relative to the start heading (+ = left). IMU if available."""
        if self.have_imu:
            return wrap(self.imu_yaw - self.imu_yaw0)
        return wrap(self.yaw - self.nav.yaw0)

    def color_cb(self, msg):
        if msg.encoding not in ('rgb8', 'bgr8'):
            return
        d, swap = msg.data, msg.encoding == 'bgr8'
        h, w = msg.height, msg.width
        first_row = h - GREEN_ROWS            # only the rows watching the
        row_green = 0                         # floor right in front
        for j in range(max(0, first_row), h):
            base = j * msg.step
            hits = 0
            for i in range(w):
                p = base + 3 * i
                r, g, b = d[p], d[p + 1], d[p + 2]
                if swap:
                    r, b = b, r
                if g > 100 and g > 1.6 * r and g > 1.6 * b:
                    hits += 1
            if hits >= 2:
                row_green += 1
        self.green_frames = self.green_frames + 1 if row_green >= GREEN_ROW_MIN else 0

    def start_cb(self, msg):
        if not msg.data:
            return
        if self.state != 'IDLE':
            return
        if not self.have_odom or not self.have_tof:
            self.get_logger().warn(
                'Odometry/TOF data not ready yet, cannot start.')
            return
        self.nav.reset(self.x, self.y, self.yaw)
        self.imu_yaw0 = self.imu_yaw
        self.green_frames = 0
        # note: servo_hist (actual joint angles) is time-series, keyed by sim
        # time - it must NOT be cleared on START
        self.get_logger().info(
            f'Heading source: {"IMU" if self.have_imu else "odom (no IMU data!)"}')
        self.start_time = self.get_clock().now()
        self.state = 'RUNNING'
        self.get_logger().info('START! Servo sweeping, going from start line to finish line.')

    # ---------- helpers ----------
    def elapsed(self):
        return (self.get_clock().now() - self.start_time).nanoseconds / 1e9

    def publish_trail(self):
        if self.elapsed() - self.last_path_t < 0.1:
            return
        self.last_path_t = self.elapsed()
        pose = PoseStamped()
        pose.header.frame_id = 'odom'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = self.x
        pose.pose.position.y = self.y
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0
        self.path.poses.append(pose)
        if len(self.path.poses) > 3000:
            self.path.poses = self.path.poses[::2]
        self.path_pub.publish(self.path)

    # ---------- main loop ----------
    def control_loop(self):
        t_sec = self.get_clock().now().nanoseconds / 1e9

        # servo command: while RUNNING the navigator sweeps the head; in every
        # other state hold it straight ahead. Publishing here ALWAYS matters:
        # TOF readings are tagged with the last published angle, so without
        # these commands no reading can ever be accepted -> START would be
        # refused forever.
        if self.state == 'RUNNING':
            angle = self.nav.servo_cmd(self.elapsed())
        else:
            angle = 0.0
        traj = JointTrajectory()
        traj.header.frame_id = 'world'        # accepted reference by the plugin
        traj.joint_names = [SERVO_JOINT]
        pt = JointTrajectoryPoint()
        pt.positions = [float(angle)]
        pt.time_from_start = Duration(seconds=0.02).to_msg()
        traj.points = [pt]
        self.servo_pub.publish(traj)
        self.servo_hist.append((t_sec, angle))

        if self.state != 'RUNNING':
            return

        cmd = Twist()
        self.publish_trail()

        if self.green_frames >= GREEN_FRAMES:
            self.state = 'FINISHED'
            self.get_logger().info(
                f'FINISH LINE detected by camera in {self.elapsed():.1f} s!')
            self.pub.publish(cmd)
            return
        if USE_ODOM_FALLBACK and self.nav.progress(self.x, self.y) >= 2.50:
            self.state = 'FINISHED'
            self.get_logger().error(
                'Passed the finish line but the camera did not see it!')
            self.pub.publish(cmd)
            return

        # 1) drive decision (keyword args: step() keeps the old sensor params)
        v, w = self.nav.step(x=self.x, y=self.y, rel=self.heading(),
                             t=self.elapsed())
        self.get_logger().info(
            f'{self.nav.state:10s} rel={math.degrees(self.heading()):6.1f}deg '
            f'target={math.degrees(self.nav.target):5.1f} blocked={self.nav.blocked}',
            throttle_duration_sec=1.0)
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = AutonomousController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
