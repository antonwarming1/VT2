This is the full dataset from the screwdriving cell project at AAU university by Ozren Vucicevic.

The raw data collected is placed in 3 folders called intrinsic, task and extrinsic data. The storage is explained in picture called "Storage and labels". File "Labels_names" contains the ID of the screws, with the label which belongs to it. Screw ID contains the metadata, and picture "Storage and labels" shows how to ID the specific screwdriving data. Picture "Dataset distribution" shows the distribution of the classes in the dataset.

The "dataset.h5" file contains the combined data from the intrinsic and task data, the key for unpacking is : key='dataset'


Intrinisc data represents the data recieved from the sensors on the screwdriver:
-Nset (1/min) -- rotation of the screwdriver per minute
-Torque (Nm) -- torque value of the screwdriver
-Current (V) -- current to the screwdriver	
-Angle (deg) -- rotation angle of the screwdriver
-Depth (mm) -- depth of the screwdriver

Task data represent the data recieved from the UR10 robot:
-TCP_x (mm) -- offset from the Tool Center Point at the starting screwdriving position in x-axis
-TCP_y (mm) -- offset from the Tool Center Point at the starting screwdriving position in y-axis
-TCP_z (mm) -- offset from the Tool Center Point at the starting screwdriving position in z-axis
-TCP_rx (rad) -- rotation components of the Tool Center Point with respect to the robot base coordinate system in x-axis
-TCP_ry (rad) -- rotation components of the Tool Center Point with respect to the robot base coordinate system in y-axis
-TCP_rz (rad) -- rotation components of the Tool Center Point with respect to the robot base coordinate system in z-axis
-Robot_I (A) -- current to the robot


Extrinsic data represents the data recieved by the microphone. Extrinsic data (clean) represents the preprocessed audio files, with the code provided called "Noise reduction".