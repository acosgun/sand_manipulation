cd ~/andrea_sand_data/ros_ws/src/sandman
source ../../devel/setup.zsh

roslaunch andrea_sand launch_camera.launch

roslaunch sandman sand_contours.launch
python sand_texture.py

roslaunch kinova_bringup kinova_robot.launch kinova_robotType:=m1n6s200

python push_executer.py
python tap_executer.py

roslaunch launch/run_net.launch
python tap_methods.py

python command_generator.py