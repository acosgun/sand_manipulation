Disconnect Doug's realsense and plug in your realsense.

Open 2 terminals.

In the first terminal, type the following commands:

    cd andrea_sand_data/ros_ws/
    . devel/setup.bash (if zsh then setup.zsh or source ../../devel/setup.zsh)
    roslaunch andrea_sand launch_camera.launch

This will launch the camera and should show the rgb and depth images.

In the second terminal, type the following commands:

    cd andrea_sand_data/ros_ws/
    . devel/setup.bash

When you are ready to record, type the following command:

    roslaunch andrea_sand record_data.launch

To stop recording, type:

    Ctrl + c

The bag file will have been recorded to andrea_sand_data/data.
To export the image files, in the second terminal type:

    cd ../data

Note the name of the bag file you wish to export, e.g. recording_2018-01-05-09-19-38.bag
To export the files, type:

    rosrun andrea_sand export_images.py <bag file name>

The files will be exported into a folder called the same as the bag file.
