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

python sand_texture.py

4) Initialize Robot:

roslaunch kinova_bringup kinova_robot.launch kinova_robotType:=m1n6s200

5) First Home the Robot to Kinova Home position via the "joystick"

6) Visual Servoing

python tap_executer.py

QUT: Set robot_type to string "kinova"
Monash: Set robot type to string "ur5"

***If you get the error "sandman.msg cannot be found". 
Monash:
go to catkin workspace main folder (~/catkin_ws for me), and run "$ . devel/setup.bash"

QUT: 
go to ~/andrea_sand_data/ros_ws/ and run catkin_make then source devel/setup.zsh

7) Train ANN or Poly Regression

(First set 'train_polyreg' variable True or False in src/train_akan_polyreg_ann.py)

python train_akan_polyreg_ann.py

9) Run alternative methods

python tap_methods.py

*If you are having issues with running keras neural net
sudo pip install tensorflow
sudo pip install keras

10) Run command generator

python command_generator.py

11) force control
rosservice call /'m1n6s200_driver'/in/start_force_control
rosservice call /'m1n6s200_driver'/in/stop_force_control
