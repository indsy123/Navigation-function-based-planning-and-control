# Navigation-function-based-planning-and-control

Dependancies
This is the code for constructing 3D navigation function over the squircle world that can be used to plan the motion of 
multiple quadrotors simulteneously. Successful running this code depends on: 

i) rotors_simulator https://github.com/ethz-asl/rotors_simulator

ii) turtlebot_gazebo from "Turtlebot2". The package is not added but can be installed using guidelines from 
http://wiki.ros.org/turtlebot/Tutorials/indigo/Turtlebot%20Installation

iii) pyquaternion: install using "pip install pyquaternion", used in controller code 

iv) the optimization was solved using academic license of gurobipy and it needs to be installed by downloading the software 
from https://www.gurobi.com/registration/download-reg and following installation instructions. 

Installation instructions: 
clone the files in your catkin_ws and run 
cd ~catkin_ws
catkin build

dependancies for installing rotors_simulator can be found at: 
https://github.com/ethz-asl/rotors_simulator

Lauching the code:
After cloing the original rotors_simulator package, change the rotors_gazebo folder from this repository
install rotors_simulator, turtlebot and planning and control 
launch quadrotors: roslaunch rotors_gazebo three_multicopters.launch world_name:=basic_R2
lauch turtlebot: roslaunch rotors_gazebo three_multicopters.launch world_name:=basic_R2
launch simulation: roslaunch planning_and_control multiple_firely.launch 

Basic Uses: 
The file "3DNF_MRSpaper_ros_R1.py" is the navigation function based plannner, define the obstacles as: 
	self.no_of_obstacles = 9 # this number includes workspace boundary
        self.O= [\
                 [0.0, 0.0, 15.0, 15.0, 0.9999, 0.0, 1.0, 1.0, 1.0], \
                 [-6.0, -8.0, 4.05, 0.8, 0.9999, 0.0, 1.0, 1.0, 0.2], \
                 [6.0, -8.0, 1.05, 0.8, 0.9999, 0.0, 1.0, 1.0, 0.8], \
                 [0.0, -4.0, 2.05, 0.8, 0.9999, 0.0, 1.0, 1.0, 0.4], \
                 [-6.0, 0.0, 1.55, 0.8, 0.9999, 0.0, 1.0, 1.0, 0.533], \
                 [6.0, 0.0, 2.55, 0.8, 0.9999, 0.0, 1.0, 1.0, 0.32], \
                 [0.0, 4.0, 1.55, 0.8, 0.9999, 0.0, 1.0, 1.0, 0.533], \
                 [-6.0, 8.0, 3.05, 0.8, 0.9999, 0.0, 1.0, 1.0, 0.266], \
                 [6.0, 8.0, 1.05, 0.8, 0.9999, 0.0, 1.0, 1.0, 0.8] \
                 ]

First obstacle is the workspace, the sequence is:
x, y, z position of obstacle center, radius of the sqauircle, squircle smoothing factor s (close to 1)
angle about z axis and scale in 3 directions
it is possible to add or remove any obstacles as per your workspace but it would need turning the navitaion 
function parameters self.k, self.lambda_sq, self.lambda_sp in the same file. 

It is possible to change the motion of the turtlebot in move_turtlebot.py file.
It is possible to add more quadrotors in launch files "three_multicopters.launch" and "multiple_firely.launch" 

The file "sensor_placement_using_qcp_ros.py" generates the goal for the quads solving qp, if you change the 
number of quads this file should be changed.



