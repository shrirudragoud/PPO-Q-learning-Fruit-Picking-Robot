#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64MultiArray.h>
#include <termios.h>
#include <stdio.h>

class KeyboardControl {
public:
    KeyboardControl() {
        // Initialize ROS node
        cmdVelPub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
        jointCmdPub = nh.advertise<std_msgs::Float64MultiArray>("/joint_commands", 1);

        // Get parameters
        nh.param("linear_scale", linear_scale, 0.5);
        nh.param("angular_scale", angular_scale, 1.0);

        // Initialize joint states
        joint_positions = {0.0, 0.0, 0.0, 0.0, 0.0}; // shoulder, elbow, wrist, gripper
        ROS_INFO("Keyboard control node initialized.");
        printInstructions();
    }

    void printInstructions() {
        printf("\n---------------------------\n");
        printf("Robot Control Keys:\n");
        printf("---------------------------\n");
        printf("Movement:\n");
        printf("   W/S : forward/backward\n");
        printf("   A/D : turn left/right\n");
        printf("\nArm Control:\n");
        printf("   I/K : shoulder up/down\n");
        printf("   O/L : elbow up/down\n");
        printf("   P/; : wrist up/down\n");
        printf("\nGripper:\n");
        printf("   Space : open/close\n");
        printf("\nOther:\n");
        printf("   Q : quit\n");
        printf("   R : reset position\n");
        printf("---------------------------\n");
    }

    int getch() {
        static struct termios oldt, newt;
        tcgetattr(STDIN_FILENO, &oldt);
        newt = oldt;
        newt.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);
        int c = getchar();
        tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
        return c;
    }

    void spin() {
        char c;
        geometry_msgs::Twist twist;
        std_msgs::Float64MultiArray joint_cmd;

        while (ros::ok()) {
            c = getch();

            // Reset commands
            twist.linear.x = 0.0;
            twist.angular.z = 0.0;
            bool update_joints = false;

            switch(c) {
                // Movement controls
                case 'w':
                    twist.linear.x = linear_scale;
                    break;
                case 's':
                    twist.linear.x = -linear_scale;
                    break;
                case 'a':
                    twist.angular.z = angular_scale;
                    break;
                case 'd':
                    twist.angular.z = -angular_scale;
                    break;

                // Arm controls
                case 'i':
                    joint_positions[0] += 0.1;
                    update_joints = true;
                    break;
                case 'k':
                    joint_positions[0] -= 0.1;
                    update_joints = true;
                    break;
                case 'o':
                    joint_positions[1] += 0.1;
                    update_joints = true;
                    break;
                case 'l':
                    joint_positions[1] -= 0.1;
                    update_joints = true;
                    break;
                case 'p':
                    joint_positions[2] += 0.1;
                    update_joints = true;
                    break;
                case ';':
                    joint_positions[2] -= 0.1;
                    update_joints = true;
                    break;

                // Gripper control
                case ' ':
                    joint_positions[3] = joint_positions[3] > 0.01 ? 0.0 : 0.03;
                    joint_positions[4] = joint_positions[4] > 0.01 ? 0.0 : 0.03;
                    update_joints = true;
                    break;

                // Reset
                case 'r':
                    std::fill(joint_positions.begin(), joint_positions.end(), 0.0);
                    update_joints = true;
                    break;

                // Quit
                case 'q':
                    ROS_INFO("Quitting...");
                    return;
            }

            // Publish movement command
            cmdVelPub.publish(twist);

            // Publish joint commands if needed
            if (update_joints) {
                joint_cmd.data = joint_positions;
                jointCmdPub.publish(joint_cmd);
            }

            ros::spinOnce();
        }
    }

private:
    ros::NodeHandle nh;
    ros::Publisher cmdVelPub;
    ros::Publisher jointCmdPub;
    double linear_scale;
    double angular_scale;
    std::vector<double> joint_positions;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "keyboard_control_node");
    KeyboardControl control;
    control.spin();
    return 0;
}