QUT:

1) Run Setup.zsh

cd ~/andrea_sand_data/ros_ws/src/sandman
source ../../devel/setup.zsh

2) Run Camera

QUT:
roslaunch andrea_sand launch_camera.launch

Monash:
rosrun cv_camera cv_camera_node /cv_camera/image_raw:=/camera/rgb/image_raw

3) Run Contour Detection 

roslaunch sandman sand_contours.launch

4) Initialize Robot:

roslaunch kinova_bringup kinova_robot.launch kinova_robotType:=m1n6s200

5) First Home the Robot to Kinova Home position via the "joystick"

6) Take the Robot to intermediate Home position:

if tool == "straight":
	rosrun kinova_demo joints_action_client.py -v m1n6s200 degree -- 379.8 220.15 54.43 162.54 112.29 22.43

else:
	rosrun kinova_demo joints_action_client.py -v m1n6s200 degree -- 371.0 210.0 50.0 179.0 58.0 43.0

7) Visual Servoing

roslaunch sandman push_executer.launch

QUT: Set robot_type to string "kinova"
Monash: Set robot type to string "ur5"

***If you get the error "sandman.msg cannot be found". 
Monash:
go to catkin workspace main folder (~/catkin_ws for me), and run "$ . devel/setup.bash"

QUT: 
go to ~/andrea_sand_data/ros_ws/ and run catkin_make then source devel/setup.zsh

8) Train ANN or Poly Regression

(First set 'train_polyreg' variable True or False in src/train_akan_polyreg_ann.py)

python train_akan_polyreg_ann.py
other ANN are in sandman.py:
python train_net_v1.py
python train_net_v2.py
python train_net_conv1.py


9) Run ANN

roslaunch sandman run_net.launch

*If you are having issues with running keras neural net
sudo pip install tensorflow
sudo pip install keras

10) Run command generator

python command_generator.py

11) force control
rosservice call /'m1n6s200_driver'/in/start_force_control
rosservice call /'m1n6s200_driver'/in/stop_force_control

12) tensorboardx

tensorboard --logdir=runs
