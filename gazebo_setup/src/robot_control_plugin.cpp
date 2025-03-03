#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <geometry_msgs/Twist.h>
#include <thread>

namespace gazebo
{
  class RobotControlPlugin : public ModelPlugin
  {
    public: RobotControlPlugin() : ModelPlugin()
    {
      // Initialize ROS
      if (!ros::isInitialized())
      {
        int argc = 0;
        char **argv = NULL;
        ros::init(argc, argv, "robot_control_plugin",
            ros::init_options::NoSigintHandler);
      }
    }

    public: void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/)
    {
      // Store the pointer to the model
      this->model = _parent;

      // Store the pointers to the joints
      this->joints["front_left_wheel"] = this->model->GetJoint("front_left_wheel_joint");
      this->joints["front_right_wheel"] = this->model->GetJoint("front_right_wheel_joint");
      this->joints["rear_left_wheel"] = this->model->GetJoint("rear_left_wheel_joint");
      this->joints["rear_right_wheel"] = this->model->GetJoint("rear_right_wheel_joint");
      this->joints["shoulder"] = this->model->GetJoint("shoulder_joint");
      this->joints["elbow"] = this->model->GetJoint("elbow_joint");
      this->joints["wrist"] = this->model->GetJoint("wrist_pitch_joint");
      this->joints["gripper_left"] = this->model->GetJoint("left_finger_joint");
      this->joints["gripper_right"] = this->model->GetJoint("right_finger_joint");

      // Create ROS node
      this->rosNode.reset(new ros::NodeHandle("robot_control"));

      // Subscribe to command topics
      this->rosSub = this->rosNode->subscribe("cmd_vel", 1,
          &RobotControlPlugin::OnCmdVel, this);
      
      this->jointSub = this->rosNode->subscribe("joint_commands", 1,
          &RobotControlPlugin::OnJointCommand, this);

      // Listen to the update event (broadcast every simulation iteration)
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&RobotControlPlugin::OnUpdate, this));

      ROS_INFO("Robot control plugin loaded!");
    }

    public: void OnUpdate()
    {
      // Apply joint commands
      for (auto const& joint : this->joints)
      {
        if (this->jointCommands.find(joint.first) != this->jointCommands.end())
        {
          joint.second->SetForce(0, this->jointCommands[joint.first]);
        }
      }

      // Apply velocity commands
      double left_velocity = (this->linear_vel - this->angular_vel * 0.5) * 10.0;
      double right_velocity = (this->linear_vel + this->angular_vel * 0.5) * 10.0;

      this->joints["front_left_wheel"]->SetVelocity(0, left_velocity);
      this->joints["rear_left_wheel"]->SetVelocity(0, left_velocity);
      this->joints["front_right_wheel"]->SetVelocity(0, right_velocity);
      this->joints["rear_right_wheel"]->SetVelocity(0, right_velocity);
    }

    private: void OnCmdVel(const geometry_msgs::Twist::ConstPtr& msg)
    {
      this->linear_vel = msg->linear.x;
      this->angular_vel = msg->angular.z;
    }

    private: void OnJointCommand(const std_msgs::Float64MultiArray::ConstPtr& msg)
    {
      // Update joint commands
      std::vector<std::string> joint_names = {
        "shoulder", "elbow", "wrist", "gripper_left", "gripper_right"
      };
      
      for (size_t i = 0; i < msg->data.size() && i < joint_names.size(); ++i)
      {
        this->jointCommands[joint_names[i]] = msg->data[i];
      }
    }

    private:
      physics::ModelPtr model;
      std::map<std::string, physics::JointPtr> joints;
      std::map<std::string, double> jointCommands;
      double linear_vel = 0;
      double angular_vel = 0;
      
      std::unique_ptr<ros::NodeHandle> rosNode;
      ros::Subscriber rosSub;
      ros::Subscriber jointSub;
      event::ConnectionPtr updateConnection;
  };

  // Register this plugin with the simulator
  GZ_REGISTER_MODEL_PLUGIN(RobotControlPlugin)
}