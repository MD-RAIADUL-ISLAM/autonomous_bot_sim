#!/usr/bin/env python3
"""Spawns 8 obstacles at random places every time it runs."""

import random
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from gazebo_msgs.srv import SpawnEntity
from std_msgs.msg import Bool

from autonomous_bot_sim.arena_layout import (
    NUM_OBSTACLES, OBS_SIZE, OBS_HEIGHT, MIN_GAP, pick_positions)

# TRANSIENT_LOCAL = latched: the controller receives the flag even if it
# connects AFTER the publish (late-joiner safe).
LATCH_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)


class ObstacleRandomizer(Node):
    def __init__(self):
        super().__init__('obstacle_randomizer')
        self.declare_parameter('seed', -1)     # -1 = different layout every run
        self.client = self.create_client(SpawnEntity, '/spawn_entity')
        self.ready_pub = self.create_publisher(Bool, '/obstacles_ready', LATCH_QOS)
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /spawn_entity service in Gazebo...')

    def make_sdf(self, name):
        s, h = OBS_SIZE, OBS_HEIGHT
        return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry><box><size>{s} {s} {h}</size></box></geometry>
      </collision>
      <visual name="visual">
        <geometry><box><size>{s} {s} {h}</size></box></geometry>
        <material>
          <ambient>0.8 0.1 0.1 1</ambient>
          <diffuse>0.8 0.1 0.1 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""

    def spawn_all(self):
        seed = self.get_parameter('seed').value
        rng = random.Random(None if seed < 0 else seed)
        positions = pick_positions(rng, NUM_OBSTACLES)
        self.get_logger().info(
            f'{NUM_OBSTACLES} obstacles, min gap between obstacles = {MIN_GAP:.2f} m')

        futures = []
        for i, (x, y) in enumerate(positions, start=1):
            req = SpawnEntity.Request()
            req.name = f'obstacle_{i}'
            req.xml = self.make_sdf(req.name)
            req.robot_namespace = ''
            # Position is set ONLY here (not in the SDF) to avoid double offset
            req.initial_pose.position.x = float(x)
            req.initial_pose.position.y = float(y)
            req.initial_pose.position.z = OBS_HEIGHT / 2.0
            futures.append((req.name, x, y, self.client.call_async(req)))

        for name, x, y, future in futures:
            rclpy.spin_until_future_complete(self, future)
            result = future.result()
            if result is not None and result.success:
                self.get_logger().info(f'Spawned {name} at X={x:.2f}, Y={y:.2f}')
            else:
                self.get_logger().error(f'Failed to spawn {name}')

	# latched "arena is complete" flag for the controller
        msg = Bool()
        msg.data = True
        self.ready_pub.publish(msg)
        self.get_logger().info('All obstacles ready -> /obstacles_ready = TRUE (latched).')
        # keep spinning briefly so the latched message is actually sent out
        end = time.monotonic() + 0.5
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleRandomizer()
    node.spawn_all()
    node.get_logger().info('All obstacles ready. Press ENTER in start_button terminal to run.')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
