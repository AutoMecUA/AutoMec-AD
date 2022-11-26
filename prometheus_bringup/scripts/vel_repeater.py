#!/usr/bin/env python3

from typing import Any
from geometry_msgs.msg._Twist import Twist


def modelSteeringCallback(message, config):
    config['steering'] = message.data

def main():
    config: dict[str, Any] = dict(cmd_vel=None)

    # Defining starting values
    config["twist"] = Twist()


if __name__ == '__main__':
    main()