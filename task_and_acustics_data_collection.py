"""
task_and_acustics_data_collection.py
=====================================
This module is the main data-collection script for the screwing-cell test rig.
It synchronises three concurrent data streams:

  1. **Task data (robot kinematics)** – TCP position and orientation (x, y, z, rx, ry,
     rz) plus robot joint current are read from a UR10 robot controller via Modbus TCP
     at approximately 400 Hz.

  2. **Acoustic data** – raw audio from a microphone is captured in a background thread
     using PyAudio and saved as a WAV file.

  3. **PLC trigger signal** – a Siemens S7 PLC (connected via Snap7) provides a Boolean
     flag that marks the start and end of each screw-driving cycle.  The flag is polled
     in the main loop; when it goes high the recording begins and when it returns low the
     data are saved to disk.

Additionally a lightweight UDP socket server (`TaskSoundServer`) allows an external GUI
application to push metadata (wood-number, process type, date and sample counter) to this
process at runtime without interrupting the recording loop.

Output files
------------
For every completed screw-driving cycle a CSV file (robot kinematics) and a WAV file
(audio) are written to  ``./data/<date><wood>/``  and mirrored to ``./dashboard/`` for
real-time visualisation.

Dependencies
------------
  pandas, numpy, pyModbusTCP, snap7, pyaudio, wave, matplotlib, socket, threading
"""

from collections.abc import Callable, Iterable, Mapping
from typing import Any
import pandas as pd
from datetime import datetime
from numpy import uint
from pyModbusTCP.client import ModbusClient
from pyModbusTCP import utils
import struct
import math
import os, time
import snap7
from snap7 import util
from snap7.types import *
from snap7.util import *
import pyaudio
import wave
import threading
import time
import matplotlib.pyplot as plt
import socket




class TaskSoundServer(threading.Thread):
    """
    A lightweight UDP server that listens for metadata messages sent by the
    external Web GUI.

    The GUI sends a comma-separated string of the form::

        "<counter>,<wood>,<process>,<date>"

    The server stores the most-recently received message and exposes it
    through :meth:`reedgui`.  Because the server runs as a daemon thread it
    shuts down automatically when the main process exits.

    Attributes
    ----------
    flag : bool
        Set to ``True`` whenever a new message has arrived and not yet been
        consumed by :meth:`reedgui`.
    data : str or int
        The raw message string from the last UDP packet, or ``0`` if no
        packet has been received yet.
    """

    def __init__(self):
        threading.Thread.__init__(self)
        self.daemon = True
        HOST = "127.0.0.1"  # The server's hostname or IP address
       # HOST = "192.168.1.100"  # The server's hostname or IP address
        
        PORT = 6000 # The port used by the server
        #s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s = socket.socket(type=socket.SOCK_DGRAM)
        self.s.bind((HOST, PORT))
        self.flag = False
        self.data = 0
    
    def run(self):
        """
        Main thread loop.  Blocks on ``socket.recv()`` waiting for incoming
        UDP datagrams.  When a datagram arrives its payload is decoded as
        UTF-8 and stored in ``self.data``.  The ``flag`` attribute is then
        raised so that the main loop knows fresh data is available.
        """
        while True:

            # Establish connection with client.
              
            self.data = self.s.recv(1024).decode()
            #print(self.data)
            if  self.data != 0:
                #print(self.data)
                self.flag = True
                
                


    def reedgui(self):
        """
        Consume and return the latest GUI message.

        Resets the internal flag after reading so that the same message is
        not returned more than once.

        Returns
        -------
        str
            The raw comma-separated metadata string if a new message has
            arrived since the last call.
        int
            ``0`` if no new message is available.
        """
        if self.flag:
            self.flag = False
            return self.data
        
        else:
            return 0

#wood = input("Enter wood number: ")
wood = 10
#process = input("Enter the process type: ")
process = 'A'
today = datetime.today().strftime('%d%m%Y')





# Function that provides the signal for the start of the screwdriving and connects to modbus_______________

def PLCsignal(db_number, start_offset, bit_offset):
    """
    Read a single Boolean bit from a Siemens S7 PLC data block.

    This function is used to detect the start of a screw-driving cycle.
    The PLC sets the specified bit high when the screwdriver is active and
    low again when the cycle has finished.

    Parameters
    ----------
    db_number : int
        The number of the S7 data block to read from (e.g. ``19``).
    start_offset : int
        Byte offset inside the data block at which the target byte is
        located (e.g. ``0`` for the first byte).
    bit_offset : int
        Bit position within the byte (0-7).  ``0`` is the most-significant
        bit in Siemens notation.

    Returns
    -------
    bool
        ``True`` if the bit is set (screw-driving in progress),
        ``False`` otherwise.
    """

    reading = client.db_read(db_number, start_offset, 1)
    a = snap7.util.get_bool(reading, 0, bit_offset)
    #print('DB Number: ' + str(db_number) + ' Bit: ' + str(start_offset) + '.' + str(bit_offset) + ' Value: ' + str(a))
    return a


try:
    client = snap7.client.Client()
    client.connect('172.20.1.148', 0, 1)
    db_number = 19
    start_offset = 0
    bit_offset = 0

    if client.get_connected():
        print("connected")
    else:
        print("could not connect to PLC")

except:
    print("could not connect to PLC")



#_________________________________________________________________________________________________________



class ModbusReader(threading.Thread):
    """
    Background thread that continuously polls the UR10 robot controller
    over Modbus TCP and caches the most recent values for all configured
    registers.

    The thread runs as a daemon so it is cleaned up automatically when the
    main process exits.  Register values are accessed from outside the
    thread via :meth:`get_register_values`.

    Parameters
    ----------
    host : str
        IP address of the robot controller (e.g. ``'172.20.1.50'``).
    port : int
        Modbus TCP port on the robot controller (typically ``502``).
    registers : dict
        Mapping from a human-readable name to the holding-register address
        on the controller.  Expected keys: ``TCP_x``, ``TCP_y``, ``TCP_z``,
        ``TCP_rx``, ``TCP_ry``, ``TCP_rz``, ``Robot_I``.

    Attributes
    ----------
    register_values : dict
        Latest raw (integer) values read from each register.
    """

    def __init__(self, host, port, registers):
        threading.Thread.__init__(self)
        self.daemon = True
        self.c = ModbusClient(host=host, port=port, auto_open=True, debug=False)
        self.registers = registers
        self.register_values = {}

        self.t_modbus = 0
        self.tflag = False

    def run(self):
        """
        Main thread loop.  Reads all configured holding registers from the
        robot controller in each iteration and updates ``self.register_values``.
        Any communication error is caught and printed so the loop can
        continue without crashing.
        """
        while True:
            try:
                
                # if self.tflag:
                #     self.t_modbus += 1
                # Read the values of the specified registers from the UR10
                reg_TCP_x = self.c.read_holding_registers(self.registers['TCP_x'])
                reg_TCP_y = self.c.read_holding_registers(self.registers['TCP_y'])
                reg_TCP_z = self.c.read_holding_registers(self.registers['TCP_z'])
                reg_TCP_rx = self.c.read_holding_registers(self.registers['TCP_rx'])
                reg_TCP_ry = self.c.read_holding_registers(self.registers['TCP_ry'])
                reg_TCP_rz = self.c.read_holding_registers(self.registers['TCP_rz'])
                reg_Robot_I = self.c.read_holding_registers(self.registers['Robot_I'])

                # Store the register values in a dictionary
                self.register_values = {
                    'TCP_x': reg_TCP_x[0],
                    'TCP_y': reg_TCP_y[0],
                    'TCP_z': reg_TCP_z[0],
                    'TCP_rx': reg_TCP_rx[0],
                    'TCP_ry': reg_TCP_ry[0],
                    'TCP_rz': reg_TCP_rz[0],
                    'Robot_I': reg_Robot_I[0]
                }
            except:
                print("Error reading register values")


    def get_register_values(self) -> dict:
        """
        Return the most recently cached register values.

        Returns
        -------
        dict
            Dictionary whose keys are the register names (``TCP_x``, …,
            ``Robot_I``) and whose values are the raw 16-bit integers read
            from the controller.
        """
        return self.register_values
    
    def get_times(self) -> int:
        """
        Return and reset the internal Modbus sample counter.

        Used for diagnostic purposes to estimate the actual Modbus polling
        rate between two events.

        Returns
        -------
        int
            Number of successful Modbus reads since the counter was last reset.
        """
        self.tflag = False
        value = self.t_modbus
        self.t_modbus = 0
        return value

    def set_times(self):
        """Enable the internal Modbus sample counter (for diagnostic use)."""
        self.tflag = True






# Connect to Modbus
try:
    c = ModbusClient(host='172.20.1.50', port=502, auto_open=True, debug=False)
    print("connected",c.open())
except ValueError:
    print("Error with host or port params")

# Setting up the task data parameters
registers = {
    'TCP_x': 400,
    'TCP_y': 401,
    'TCP_z': 402,
    'TCP_rx': 403,
    'TCP_ry': 404,
    'TCP_rz': 405,
    'Robot_I': 450
}



# Create a ModbusReader thread and start it
modbus_reader = ModbusReader('172.20.1.50', 502, registers)
modbus_reader.start()





# Initialize the dataframe
df = pd.DataFrame(columns=['Time', 'TCP_x', 'TCP_y', 'TCP_z', 'TCP_rx', 'TCP_ry', 'TCP_rz', 'Robot_I'])



def MakeNewFolder(foldername: str) -> None:
    """
    Create a directory (and all intermediate directories) at *foldername*.

    Wraps ``os.makedirs`` and prints a status message on success or failure.
    The function does not raise an exception if creation fails.

    Parameters
    ----------
    foldername : str
        Absolute or relative path of the directory to create.
    """
    path = foldername
    try:
        os.makedirs(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)


def CheckIfFolderExist(foldername: str) -> bool:
    """
    Check whether a directory already exists on the filesystem.

    Parameters
    ----------
    foldername : str
        Path to test.

    Returns
    -------
    bool
        ``True`` if *foldername* is an existing directory, ``False`` otherwise.
    """
    isdir = os.path.isdir(foldername)
    return isdir


def unsigned(a: int) -> int:
    """
    Convert a raw Modbus 16-bit signed register value to a signed integer.

    Modbus registers are unsigned 16-bit (0–65535).  The UR10 controller
    stores signed values using two's-complement encoding, so any raw value
    greater than 32767 represents a negative number.  This function shifts
    values in the range [32768, 65535] back into the negative range
    [-32767, 0].

    Parameters
    ----------
    a : int
        Raw unsigned register value (0–65535).

    Returns
    -------
    int
        Signed integer equivalent of the register value.
    """
    if a > 32767:
        a = a - 65535
    else:
        a = a
    return a
#_______________________________________________________________________________________

""" Setting up the microphone recording class as a thread.
    It will run in the background, during the data collection.
"""


class SoundRecorderThread(threading.Thread):
    """
    Background thread that records audio from the default system microphone.

    Recording starts as soon as the thread is started (``thread.start()``) and
    continues until :meth:`stop_recording` is called.  Captured raw PCM frames
    are accumulated in memory and can be retrieved via :meth:`get_frames` for
    subsequent saving to a WAV file.

    Audio settings
    --------------
    * Sample rate  : 44 100 Hz
    * Bit depth    : 16-bit signed (``paInt16``)
    * Channels     : mono (1)
    * Buffer size  : 1 024 frames

    Attributes
    ----------
    frames : list of bytes
        Accumulated raw PCM audio chunks.
    should_stop : bool
        When set to ``True`` by :meth:`stop_recording` the recording loop
        exits on its next iteration.
    """

    def __init__(self):
        # Call the constructor of the parent class (threading.Thread)
        threading.Thread.__init__(self)
        
        # Initialize some default parameters for recording audio
        self.frames = []
        self.should_stop = False
        self.CHUNK = 1024 #buffer
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100

    def run(self):
        """
        Main thread loop.  Opens a PyAudio input stream and appends each
        chunk of raw PCM data to ``self.frames`` until ``should_stop`` is
        set.  The stream and the PyAudio context are cleanly closed before
        the thread exits.
        """
        audio = pyaudio.PyAudio()
        # Open a stream for recording audio with the default input device
        stream = audio.open(format=self.FORMAT, channels=self.CHANNELS,
                            rate=self.RATE, input=True,
                            frames_per_buffer=self.CHUNK)
        # Continuously read audio data from the stream and append it to self.frames
        while not self.should_stop:
            data = stream.read(self.CHUNK)
            self.frames.append(data)
            
        # Stop the stream and close it
        stream.stop_stream()
        stream.close()
        audio.terminate()

    def stop_recording(self) -> None:
        """
        Signal the recording thread to stop after the current buffer read.

        Sets ``should_stop`` to ``True``; the thread will finish its current
        ``stream.read()`` call and then exit the loop cleanly.
        """
        self.should_stop = True

    def get_frames(self) -> list:
        """
        Return all recorded audio frames collected so far.

        Returns
        -------
        list of bytes
            Raw PCM chunks that can be joined and written to a WAV file via
            :func:`save_recording`.
        """
        return self.frames

def save_recording(frames: list, filename: str) -> None:
    """
    Write a list of raw PCM audio chunks to a WAV file on disk.

    The function uses fixed audio parameters that match the settings used
    by :class:`SoundRecorderThread`:

    * Channels   : 1 (mono)
    * Sample width: 2 bytes (16-bit)
    * Frame rate : 44 100 Hz

    Parameters
    ----------
    frames : list of bytes
        Raw PCM data as returned by :meth:`SoundRecorderThread.get_frames`.
    filename : str
        Destination path for the WAV file (including the ``.wav`` extension).
    """
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(b''.join(frames))



#________________________________________________________________________________________
"""
Main loop that monitors the PLC signal and decides when will the data be recorded and where will it be saved.
"""

# ---------------------------------------------------------------------------
# Main-loop configuration
# ---------------------------------------------------------------------------

# Target interval between consecutive data samples (milliseconds).
# At 400 ms the effective sampling rate is approximately 2.5 Hz; however,
# the actual rate depends on the Modbus polling speed of the robot controller.
desired_frequency = 400  # 400 ms
desired_period = desired_frequency / 1000  # Convert to seconds


# Variables for the audio recorder thread
recorder = None   # SoundRecorderThread instance created fresh for each cycle
frames = []       # Accumulated PCM frames for the current recording

# Variables for the PLC signal state machine
flag = False      # True while a screw-driving cycle is in progress
counter = 1       # Monotonically increasing sample index used in file names

# Output directory: ./data/  (created automatically if it does not exist)
ext_dir = os.getcwd() + '\data'
directory = os.path.expanduser(ext_dir)

# Timing helpers used for diagnostic frequency measurements
t_array2 = []
t3 = 0
t1 = 0

# Start the UDP metadata server so the GUI can push session info at any time
server = TaskSoundServer()
server.start()
gui_info = 0

# ---------------------------------------------------------------------------
# Main polling loop
# ---------------------------------------------------------------------------
# The loop runs indefinitely.  In each iteration it:
#   1. Checks for metadata updates from the GUI (wood ID, process, date).
#   2. Reads the PLC trigger bit.
#   3. If the trigger goes high, starts a new audio recording and resets
#      the kinematic data buffer.
#   4. While the trigger is high, appends the latest Modbus register values
#      (robot TCP pose + current) to the data buffer.
#   5. When the trigger goes low, stops recording, converts the raw register
#      values to engineering units, and saves both the CSV and WAV files.
while True:
    # if counter == 2:
    #     break
    start_time_loop = time.time()
    
    # Check the PLC signal
    # if flag:
    #     t1 = time.monotonic()
    #     t_array2.append(t1-t3)
        
    #     t3 = t1
    
    
    gui_info = server.reedgui()
    if str(gui_info) != "0":
        #print(data)
        gui_info_list = gui_info.split(',')
        counter = int(gui_info_list[0])
        wood = gui_info_list[1]
        process = gui_info_list[2]
        today = gui_info_list[3]
        print("counter is set to: "+str(counter))
        print("woodnumber is set to: "+str(wood))
        print("process is set to: "+str(process))
        print("date is set to: "+str(today))
        gui_info = 0
    result = PLCsignal(db_number, start_offset, bit_offset)
    # if flag:
    #     t2 = time.monotonic()
    #     t_array.append((t2-t1))


    # If the function returns True, set the flag to True and start recording
    if result and not flag:
        ##print("Recording started")
        flag = True
        # modbus_reader.set_times()
        t2 = time.monotonic()
        start_time = datetime.now()
        #For recording task data
        data = []
        
        #Recording audio
        #recorder = SoundRecorderThread()
        #recorder.start()
    
    register_values = modbus_reader.get_register_values()
    
    # If the flag is True, record audio until the PLC signal goes back to False
    if flag:
        current_time = datetime.now()
        elapsed_time = (current_time - start_time).total_seconds() * 1000
        data.append([elapsed_time, register_values['TCP_x'], register_values['TCP_y'], register_values['TCP_z'], register_values['TCP_rx'], register_values['TCP_ry'], register_values['TCP_rz'], register_values['Robot_I']])
        
        # If the PLC signal is False or if the recording has reached its maximum duration, stop recording
        if not result:
            flag = False
            counter += 1
            # samples = modbus_reader.get_times()
            t3 = time.monotonic()
            print("Recording stopped")
            

            
            
            df = pd.DataFrame(data=data, columns=['Time', 'TCP_x', 'TCP_y', 'TCP_z','TCP_rx', 'TCP_ry', 'TCP_rz', 'Robot_I'])
            df = df.applymap(unsigned)
            df[['TCP_x', 'TCP_y', 'TCP_z']] = df[['TCP_x', 'TCP_y', 'TCP_z']] / 10
            df[['TCP_rx', 'TCP_ry', 'TCP_rz', 'Robot_I']] = df[['TCP_rx', 'TCP_ry', 'TCP_rz', 'Robot_I']] / 1000
            df = df.rename(columns={'Time': 'Time (ms)', 'TCP_x': 'TCP_x (mm)', 'TCP_y': 'TCP_y (mm)', 'TCP_z': 'TCP_z (mm)', 'TCP_rx': 'TCP_rx (mm)', 'TCP_ry': 'TCP_ry (mm)', 'TCP_rz': 'TCP_rz (mm)', 'Robot_I': 'Robot_I (A)'})

            print("elapsed time: "+str(elapsed_time))
            sf = len(df.index)/int(elapsed_time/1000)
            print("Sampling frequency is:", round(sf),"Hz")
            # print("robot frequency is: " + str(samples/((t3-t2)))+"Hz")
            print("our sampling: "+str(len(df.index)/((t3-t2)))+"Hz")
            print(elapsed_time)
            print(len(df.index))
            print(df)

            

            filename_t = os.path.join(directory+"\\"+str(today)+str(wood), f"{today}{wood}{process}{counter}")
            if not CheckIfFolderExist(directory+"\\"+str(today)+str(wood)):
                MakeNewFolder(directory+"\\"+str(today)+str(wood))
            df.to_csv(filename_t+".csv", index=False)
            
            # saving the robot data for the dashboard
            if not CheckIfFolderExist("dashboard"):
                MakeNewFolder("dashboard")
            df.to_csv("dashboard\\"+f"{today}{wood}{process}{counter}"+".csv", index=False)
       
            
            #recorder.stop_recording()
            #frames = recorder.get_frames()

            # Save the recorded audio as a WAV file
            #filename = os.path.join(directory+"\{today}{wood}", f"{today}{wood}{today}{wood}{process}{counter}.wav")
            #save_recording(frames, filename_t+".wav")

            # Saving the audio data for the dashboard
            #save_recording(frames, "dashboard\\"+f"{today}{wood}{process}{counter}"+".wav")

    # Give the loop delay to allow for the Web GUI to update
    # time.sleep(3)

            

            
# plt.plot(t_array,".")
# plt.xlabel("samples")
# plt.ylabel("seconds getting plc status")
# plt.show()

# plt.plot(t_array2[1:],".")
# plt.xlabel("samples")
# plt.ylabel("seconds getting plc status")
# plt.show()

            




    
    
    


