# Orange Harvesting Robot Architecture

```
+------------------------+
|    Robot Structure    |
+------------------------+
         |
    +----+----+
    |         |
+-------+  +-------+
| Base  |  |  Arm  |
+-------+  +-------+
|         |
| - 4WD   | - Shoulder
| - Diff  | - Elbow
| Drive   | - Wrist
|         | - Gripper
+---------+---------+

Neural Network Architecture:
===========================

Actor Network:
-------------
[Input Layer (29)] --> [BatchNorm]
        |
        v
[Hidden (58)] --> [ReLU] --> [BatchNorm]
        |
        v
[Hidden (29)] --> [ReLU] --> [BatchNorm]
        |
        v
[Output (6)] --> [Tanh]

Critic Network:
--------------
[Input Layer (29)] --> [BatchNorm]
        |
        v
[Hidden (58)] --> [ReLU] --> [BatchNorm]
        |
        v
[Hidden (29)] --> [ReLU] --> [BatchNorm]
        |
        v
[Output (1)] --> [Linear]

Training Flow:
=============
                   +----------------+
                   | Environment    |
                   +----------------+
                          ^
                          |
                    State | Reward
                          |
              +----------+----------+
              |    PPO Trainer     |
              +-------------------+
                   ^           |
                   |           v
            Value  |     +------------+
            Est.   |     |   Actor   |
                   |     +------------+
              +----+----+      |
              | Critic  |      |
              +---------+      v
                          Actions

Observation Space:
=================
+------------------+
| Robot State (9)  |
+------------------+
| - Position (2)   |
| - Orientation (2)|
| - Velocities (2) |
| - Joint Angles(3)|
+------------------+

+------------------+
| Env State (4)    |
+------------------+
| - Fruit Dist     |
| - Direction      |
| - Progress       |
| - Fruits Left    |
+------------------+

+------------------+
| LIDAR (16)       |
+------------------+
| Ray Distances    |
+------------------+

Action Space:
============
+------------------+
| Continuous (6)   |
+------------------+
| - Forward Vel    |
| - Turn Vel       |
| - Shoulder Pos   |
| - Elbow Pos      |
| - Wrist Pos      |
| - Gripper Pos    |
+------------------+

Reward Components:
================
[Distance] --> Exponential Scaling
[Progress] --> Path Efficiency
[Action]   --> Movement Smoothness
[Success]  --> Task Completion
```
