#!/usr/bin/env python3
"""ROS 2 wrapper around Navigator: sonars + odom in, /cmd_vel out."""
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Range, Image, Imu
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool

from autonomous_bot_sim.navigator import Navigator, wrap

# ---------- Settings ----------
FRONT_EXTENT = 0.10            # axle (base_link) -> front bumper/sonar
ARENA_LINES  = 2.50            # start line (y=-1.25) -> finish line (y=+1.25)
FINISH_DIST  = ARENA_LINES - FRONT_EXTENT   # bumper reaches finish line
USE_ODOM_FALLBACK = False   # True = also stop when odometry says the line was passed


LOOP_PERIOD  = 0.05            # 20 Hz


GREEN_PIXELS = 10     # min green pixels (of 256) to count a frame as "green"
GREEN_FRAMES = 2      # consecutive green frames needed (noise filter)

# Gazebo sensor/odom publishers use BEST_EFFORT. Subscribers must match.
SENSOR_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST, depth=5)
CMD_QOS = QoSProfile(depth=10)

SONARS = ['fl', 'fc', 'fr', 'l', 'r']


class AutonomousController(Node):
    def __init__(self):
        super().__init__('autonomous_controller')
        self.nav = Navigator()

        self.pub = self.create_publisher(Twist, '/cmd_vel', CMD_QOS)
        self.path_pub = self.create_publisher(Path, '/trail', 1)

        self.create_subscription(Odometry, '/odom', self.odom_cb, SENSOR_QOS)
        self.create_subscription(Bool, '/start_button', self.start_cb, CMD_QOS)
        self.dist = {k: 2.0 for k in SONARS}
        self.got = set()
        for k in SONARS:
            self.create_subscription(
                Range, f'/sonar_{k}_range',
                lambda msg, k=k: self.sonar_cb(k, msg), SENSOR_QOS)

        self.green_frames = 0
        self.create_subscription(Image, '/color_sensor/image_raw',
                                 self.color_cb, SENSOR_QOS)
        
        self.have_odom = False
        self.x = self.y = self.yaw = 0.0
        
        
        self.have_imu = False
        self.imu_yaw = self.imu_yaw0 = 0.0
        self.create_subscription(Imu, '/imu/data', self.imu_cb, SENSOR_QOS)
        
        
        self.state = 'IDLE'       # IDLE -> RUNNING -> FINISHED / TIMEOUT
        self.start_time = None
        self.path = Path()
        self.path.header.frame_id = 'odom'
        self.last_path_t = 0.0

        self.create_timer(LOOP_PERIOD, self.control_loop)
        self.get_logger().info('Controller ready. Waiting for START button...')

    # ---------- callbacks ----------
    def sonar_cb(self, key, msg):
        r = float(msg.range)
        if not math.isfinite(r) or r > msg.max_range:
            r = float(msg.max_range)          # nothing detected
        if r <= 0.0:
            return
        self.dist[key] = r
        self.got.add(key)

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
        d, swap, hits = msg.data, msg.encoding == 'bgr8', 0
        for j in range(msg.height):
            base = j * msg.step
            for i in range(msg.width):
                p = base + 3 * i
                r, g, b = d[p], d[p + 1], d[p + 2]
                if swap:
                    r, b = b, r
                if g > 100 and g > 1.6 * r and g > 1.6 * b:
                    hits += 1
        self.green_frames = self.green_frames + 1 if hits >= GREEN_PIXELS else 0

    def start_cb(self, msg):
        if not msg.data:
            return
        if self.state != 'IDLE':
            self.get_logger().warn('Already started or finished. Restart simulation to run again.')
            return
        if not self.have_odom or len(self.got) < len(SONARS):
            self.get_logger().warn('Odometry/sonar data not ready yet, cannot start.')
            return
        self.nav.reset(self.x, self.y, self.yaw)
        self.imu_yaw0 = self.imu_yaw
        self.get_logger().info(f'Heading source: {"IMU" if self.have_imu else "odom (no IMU data!)"}')
        self.start_time = self.get_clock().now()
        self.state = 'RUNNING'
        self.get_logger().info('START! Going from start line to finish line.')
    
    # ---------- helpers ----------
    def elapsed(self):
        return (self.get_clock().now() - self.start_time).nanoseconds / 1e9

    def publish_trail(self):
        t = self.elapsed()
        if t - self.last_path_t < 0.2:
            return
        self.last_path_t = t
        ps = PoseStamped()
        ps.header.frame_id = 'odom'
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y = self.x, self.y
        self.path.poses.append(ps)
        self.path.header.stamp = ps.header.stamp
        self.path_pub.publish(self.path)

    # ---------- control loop ----------
    def control_loop(self):
        cmd = Twist()
        if self.state != 'RUNNING':
            self.pub.publish(cmd)
            return

        self.publish_trail()

        if self.green_frames >= GREEN_FRAMES:
            self.state = 'FINISHED'
            self.get_logger().info(
                f'FINISH LINE detected by color sensor in {self.elapsed():.1f} s!')
            self.pub.publish(cmd)
            return
            
        if USE_ODOM_FALLBACK and self.nav.progress(self.x, self.y) >= FINISH_DIST + 0.20:
            self.state = 'FINISHED'
            self.get_logger().error('Passed the finish line but the color sensor did not see it!')
            self.pub.publish(cmd)
            return
        

        d = self.dist
        rel = self.heading()
        v, w = self.nav.step(d['fl'], d['fc'], d['fr'], d['l'], d['r'],
                             self.x, self.y, rel, self.elapsed())
        
        self.get_logger().info(
            f"{self.nav.state:10s} rel={math.degrees(rel):6.1f}deg fc={d['fc']:.2f} "
            f"blocked={self.nav.blocked}", throttle_duration_sec=1.0)
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
        if rclpy.ok():
            node.pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
