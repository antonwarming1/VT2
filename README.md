# screwingcell_task_and_acoustics


This program consistet of the two python scripts, one for the gui and screwdriver data, and one for the task and extrinsic (sound) data. Tor launch the program run the run.bat file.

Between the two scripts is a UDP connection, this make the gui able to specify what proccestype, woodnumber, date and pinhole is begin run

The robot data from the robot is pulish with around 115Hz, we are sampling with around 400-500Hz.


PC to robot communication is modbusTCP
Screwdriver to PC communication is Serial port
PC to PLC communication is IP/TCP (python snap7)





remeber to have open an running WSK3
use ./run.bat


There is a tab that can delete samples and one for editing sample. in the editing sample can be used to change the label of the sample.


##  Before start and getting data
* Open the WSK3 program fron weber and plug in the USB from the C30S controller
![image](https://user-images.githubusercontent.com/72868875/181718508-928cc301-6564-4261-9b72-ffca0b3669c4.png)
* Go to settings and select the right COM port
* Select then where the kxml files is going to be saved, chosse WS3 folder in the screwingcell_task_and_acoustics folder
![image](https://user-images.githubusercontent.com/72868875/181718796-60de4b94-4b70-4f86-abc3-4d9406893dc0.png)

