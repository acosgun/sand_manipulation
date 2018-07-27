QUT:

1) Run Setup.zsh

cd ~/andrea_sand_data/ros_ws/src/sandman
source ../../devel/setup.zsh



2) Run Camera

QUT:
roslaunch andrea_sand launch_camera.launch

Monash:
rosrun cv_camera cv_camera_node /cv_camera/image_raw:=/camera/rgb/image_raw



3) Initialize Robot:

roslaunch kinova_bringup kinova_robot.launch kinova_robotType:=m1n6s200

and bring robot to home position using the joystick



4) (IF NOT TRAINED ALREADY) Train ANN and/or Poly Regression

(First set 'train_polyreg' variable True or False in src/train_akan_polyreg_ann.py)

python train_akan_polyreg_ann.py




4) Push Action controller (sand_contour + push_executer + run_net)
roslaunch sandman push_action.launch



5) Run command generator

python command_generator.py


