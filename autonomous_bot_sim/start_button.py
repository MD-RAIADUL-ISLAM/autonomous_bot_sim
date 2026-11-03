#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


class StartButton(Node):
    def __init__(self):
        super().__init__('start_button')
        self.pub = self.create_publisher(Bool, '/start_button', 10)


def main(args=None):
    rclpy.init(args=args)
    node = StartButton()
    print('START BUTTON: ENTER chapleyi run shuru korbe (Ctrl+C = quit)')
    try:
        while rclpy.ok():
            input()
            msg = Bool()
            msg.data = True
            node.pub.publish(msg)
            print('Start signal pathano holo!')
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
