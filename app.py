import pybullet as p
import pybullet_data
import time

# Connect to the physics server
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load ground plane
planeId = p.loadURDF("plane.urdf")

# Load your robot URDF
startPos = [0, 0, 0.2]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
robotId = p.loadURDF("fruit_harvesting_robot.urdf", startPos, startOrientation)

# Get number of joints
numJoints = p.getNumJoints(robotId)
print(f"Number of joints: {numJoints}")

# Print joint information
for i in range(numJoints):
    jointInfo = p.getJointInfo(robotId, i)
    print(f"Joint {i}: {jointInfo[1].decode('utf-8')}")

# Keep the visualization open
while True:
    p.stepSimulation()
    time.sleep(0.01)