# Screwing Cell Task and Acoustics Data Collection

This system collects synchronized data from a screwing-cell test rig, capturing robot kinematics, screwdriver torque/angle data, and acoustic signals during screw-driving operations.

---

## System Communication Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              SCREWING CELL DATA COLLECTION SYSTEM                       │
└─────────────────────────────────────────────────────────────────────────────────────────┘

                        ┌──────────────────────────────────────┐
                        │         INDUSTRIAL EQUIPMENT         │
                        └──────────────────────────────────────┘

   ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
   │   UR10 ROBOT     │     │  SIEMENS S7 PLC  │     │ WEBER SCREWDRIVER│
   │   CONTROLLER     │     │                  │     │   C30S CONTROLLER│
   │                  │     │                  │     │                  │
   │  IP: 172.20.1.50 │     │ IP: 172.20.1.148 │     │   USB/Serial     │
   │  Port: 502       │     │  DB19, Bit 0.0   │     │                  │
   │                  │     │                  │     │                  │
   │  Provides:       │     │  Provides:       │     │  Provides:       │
   │  • TCP Position  │     │  • Trigger Signal│     │  • Torque Data   │
   │    (x,y,z)       │     │    (Start/Stop)  │     │  • Angle Data    │
   │  • Orientation   │     │                  │     │  • KXML Files    │
   │    (rx,ry,rz)    │     │                  │     │                  │
   │  • Joint Current │     │                  │     │                  │
   └────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
            │                        │                        │
            │ Modbus TCP             │ Snap7 (S7 Protocol)    │ Serial/USB
            │ ~400 Hz                │ Polling                │ via WSK3
            │                        │                        │
            ▼                        ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    DATA COLLECTION PC                                   │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                        task_and_acustics_data_collection.py                     │   │
│  │                              (Background Service)                               │   │
│  ├─────────────────────────────────────────────────────────────────────────────────┤   │
│  │                                                                                 │   │
│  │   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │   │
│  │   │  ModbusReader   │  │  PLCsignal()    │  │ SoundRecorder   │                │   │
│  │   │    Thread       │  │   Function      │  │    Thread       │                │   │
│  │   │                 │  │                 │  │                 │                │   │
│  │   │ Reads robot     │  │ Reads trigger   │  │ Records audio   │                │   │
│  │   │ registers       │  │ bit from PLC    │  │ via PyAudio     │                │   │
│  │   │ continuously    │  │ DB19[0].0       │  │ 44.1kHz, 16-bit │                │   │
│  │   └─────────────────┘  └─────────────────┘  └─────────────────┘                │   │
│  │                                                                                 │   │
│  │   ┌─────────────────────────────────────────────────────────────┐              │   │
│  │   │                   TaskSoundServer (UDP)                     │              │   │
│  │   │                   Listens on 127.0.0.1:6000                 │              │   │
│  │   │                   Receives: counter, wood, process, date    │              │   │
│  │   └─────────────────────────────────────────────────────────────┘              │   │
│  │                                         ▲                                       │   │
│  └─────────────────────────────────────────┼───────────────────────────────────────┘   │
│                                            │ UDP                                       │
│                                            │ Port 6000                                 │
│                                            │                                           │
│  ┌─────────────────────────────────────────┼───────────────────────────────────────┐   │
│  │                          ConverterKxmlToJson.py                                 │   │
│  │                              (GUI Application)                                  │   │
│  ├─────────────────────────────────────────────────────────────────────────────────┤   │
│  │                                                                                 │   │
│  │   ┌─────────────────────────────────────────────────────────────┐              │   │
│  │   │                      Tkinter GUI                            │              │   │
│  │   │  • Start/Stop data collection                               │              │   │
│  │   │  • Set wood number, process type, date                      │              │   │
│  │   │  • Change labels / Delete samples                           │              │   │
│  │   └─────────────────────────────────────────────────────────────┘              │   │
│  │                                                                                │   │
│  │   Uses: folderManager.py                                                       │   │
│  │   • Monitors WSK3 folder for KXML files                                        │   │
│  │   • Converts KXML → JSON                                                       │   │
│  │   • Manages folder structure and file operations                               │   │
│  │                                                                                 │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
│  ┌───────────────────────────────────────┐  ┌───────────────────────────────────────┐  │
│  │         MICROPHONE                    │  │         WSK3 SOFTWARE                 │  │
│  │  • Connected via audio interface      │  │  • Weber screwdriver monitoring       │  │
│  │  • 44.1 kHz, 16-bit, Mono             │  │  • Outputs KXML to WSK3 folder        │  │
│  └───────────────────────────────────────┘  └───────────────────────────────────────┘  │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘

                        ┌──────────────────────────────────────┐
                        │            OUTPUT FILES              │
                        └──────────────────────────────────────┘

        ┌─────────────────────────────────────────────────────────────────┐
        │                        ./data/<date><wood>/                     │
        ├─────────────────────────────────────────────────────────────────┤
        │  • {date}{wood}{process}{counter}.csv   (Robot kinematics)      │
        │  • {date}{wood}{process}{counter}.wav   (Audio recording)       │
        │  • {date}{wood}{process}{counter}.json  (Screwdriver data)      │
        └─────────────────────────────────────────────────────────────────┘
        
        ┌─────────────────────────────────────────────────────────────────┐
        │                        ./data_kxml/<date><wood>/                │
        ├─────────────────────────────────────────────────────────────────┤
        │  • {date}{wood}{process}{counter}.kxml  (Raw screwdriver data)  │
        └─────────────────────────────────────────────────────────────────┘

        ┌─────────────────────────────────────────────────────────────────┐
        │                        ./dashboard/                             │
        ├─────────────────────────────────────────────────────────────────┤
        │  • Mirror of all files for real-time visualization              │
        └─────────────────────────────────────────────────────────────────┘
```

---

## Communication Protocols Summary

| Source | Destination | Protocol | Port/Address | Data | Frequency |
|--------|-------------|----------|--------------|------|-----------|
| UR10 Robot | PC | Modbus TCP | 172.20.1.50:502 | TCP pose (x,y,z,rx,ry,rz), Joint Current | ~400-500 Hz |
| Siemens PLC | PC | Snap7 (S7) | 172.20.1.148 | Trigger signal (DB19, Byte 0, Bit 0) | Polling |
| Screwdriver C30S | PC | USB/Serial | COM Port | KXML files (via WSK3) | Per screw cycle |
| GUI → Data Script | Internal | UDP | 127.0.0.1:6000 | Metadata (counter, wood, process, date) | On user action |
| Microphone | PC | Audio Interface | PyAudio | Raw PCM audio | 44.1 kHz |

---

## Data Flow Sequence

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                         SCREW-DRIVING CYCLE DATA FLOW                             │
└───────────────────────────────────────────────────────────────────────────────────┘

  USER                GUI                DATA COLLECTION          HARDWARE
    │                  │                      │                       │
    │  1. Set params   │                      │                       │
    │─────────────────►│                      │                       │
    │                  │                      │                       │
    │  2. Click Start  │                      │                       │
    │─────────────────►│                      │                       │
    │                  │                      │                       │
    │                  │  3. UDP: metadata    │                       │
    │                  │─────────────────────►│                       │
    │                  │                      │                       │
    │                  │                      │  4. PLC signal HIGH   │
    │                  │                      │◄──────────────────────│
    │                  │                      │                       │
    │                  │                      │  ┌─────────────────┐  │
    │                  │                      │  │ START RECORDING │  │
    │                  │                      │  └─────────────────┘  │
    │                  │                      │                       │
    │                  │                      │  5. Read Modbus regs  │
    │                  │                      │◄─────────────────────►│ (UR10)
    │                  │                      │     (continuous)      │
    │                  │                      │                       │
    │                  │                      │  6. Record audio      │
    │                  │                      │◄──────────────────────│ (Mic)
    │                  │                      │     (continuous)      │
    │                  │                      │                       │
    │                  │                      │  7. PLC signal LOW    │
    │                  │                      │◄──────────────────────│
    │                  │                      │                       │
    │                  │                      │  ┌─────────────────┐  │
    │                  │                      │  │ STOP RECORDING  │  │
    │                  │                      │  │ SAVE .csv, .wav │  │
    │                  │                      │  └─────────────────┘  │
    │                  │                      │                       │
    │                  │  8. KXML file appears (via WSK3)             │
    │                  │◄─────────────────────────────────────────────│
    │                  │                      │                       │
    │                  │  ┌─────────────────────────────────────────┐ │
    │                  │  │ CONVERT KXML → JSON, SAVE TO FOLDERS    │ │
    │                  │  └─────────────────────────────────────────┘ │
    │                  │                      │                       │
    ▼                  ▼                      ▼                       ▼
```

---

## Script Responsibilities

### 1. task_and_acustics_data_collection.py
**Role:** Background data collection service

| Component | Function |
|-----------|----------|
| `ModbusReader` (Thread) | Continuously polls UR10 robot for TCP pose and joint current |
| `SoundRecorderThread` (Thread) | Records audio from microphone during screw-driving |
| `TaskSoundServer` (Thread) | UDP server receiving metadata from GUI |
| `PLCsignal()` | Reads PLC trigger bit to detect cycle start/end |
| Main Loop | Orchestrates recording based on PLC signals |

**Outputs:**
- `.csv` files with robot kinematics (Time, TCP_x/y/z, TCP_rx/ry/rz, Robot_I)
- `.wav` files with acoustic recordings

### 2. ConverterKxmlToJson.py
**Role:** GUI application and screwdriver data processor

| Component | Function |
|-----------|----------|
| `Program` class | Main application logic |
| `SystemLoop()` | Monitors WSK3 folder for new KXML files |
| UDP Client | Sends session metadata to data collection script |
| Tkinter GUI | User interface for configuration and control |

**Features:**
- Start/Stop data collection
- Set process parameters (wood number, process type, date)
- Change sample labels
- Delete samples

### 3. folderManager.py
**Role:** File system utility class

| Method | Function |
|--------|----------|
| `SaveJsonFile()` | Saves converted JSON to data folders |
| `SaveKxmlFile()` | Archives original KXML files |
| `ConvertKxmlToJson()` | Parses KXML using xmltodict |
| `Wsk3FolderDetect()` | Monitors folder for new KXML files |
| `DeleteFileJsonAndKxml()` | Removes sample files |
| `ChangeLabel()` | Renames files with new process label |

---

## Code Connection Details

This section explains exactly where each communication connection is established in the code and what happens at those points.

### 1. PLC Connection (Siemens S7 via Snap7)

**File:** `task_and_acustics_data_collection.py`  
**Lines:** 168-181

```python
# Connection setup
client = snap7.client.Client()
client.connect('172.20.1.148', 0, 1)  # IP, rack, slot
db_number = 19      # Data Block number
start_offset = 0    # Byte offset in DB
bit_offset = 0      # Bit position (0-7)
```

**Reading the trigger signal:**  
**Lines:** 139-163 (`PLCsignal` function)

```python
def PLCsignal(db_number, start_offset, bit_offset):
    reading = client.db_read(db_number, start_offset, 1)  # Read 1 byte from DB19
    a = snap7.util.get_bool(reading, 0, bit_offset)       # Extract bit 0
    return a  # True = screwing in progress, False = idle
```

**What happens:** The PLC stores a Boolean flag in Data Block 19, Byte 0, Bit 0. When the screwdriver starts, the PLC sets this bit HIGH. When screwing finishes, it goes LOW. The main loop polls this continuously to know when to start/stop recording.

---

### 2. Robot Connection (UR10 via Modbus TCP)

**File:** `task_and_acustics_data_collection.py`  
**Lines:** 276-282 (connection) and 186-268 (ModbusReader class)

```python
# Initial connection test
c = ModbusClient(host='172.20.1.50', port=502, auto_open=True, debug=False)

# Register addresses for robot data
registers = {
    'TCP_x': 400,    # X position (mm * 10)
    'TCP_y': 401,    # Y position
    'TCP_z': 402,    # Z position
    'TCP_rx': 403,   # Rotation X (rad * 1000)
    'TCP_ry': 404,   # Rotation Y
    'TCP_rz': 405,   # Rotation Z
    'Robot_I': 450   # Joint current (A * 1000)
}

# Start background reader thread
modbus_reader = ModbusReader('172.20.1.50', 502, registers)
modbus_reader.start()
```

**Continuous reading in ModbusReader.run():**  
**Lines:** 221-248

```python
def run(self):
    while True:
        # Read each register from the robot controller
        reg_TCP_x = self.c.read_holding_registers(self.registers['TCP_x'])
        reg_TCP_y = self.c.read_holding_registers(self.registers['TCP_y'])
        # ... (all 7 registers)
        
        # Cache values for main loop to access
        self.register_values = {
            'TCP_x': reg_TCP_x[0],
            'TCP_y': reg_TCP_y[0],
            # ...
        }
```

**What happens:** The `ModbusReader` thread runs continuously in the background, polling the robot controller at ~400 Hz. It reads 7 holding registers containing the TCP (Tool Center Point) position, orientation, and joint current. The main loop calls `get_register_values()` to get the latest cached values without blocking. 


---

### 3. UDP Communication (GUI → Data Collection)

**Server side in** `task_and_acustics_data_collection.py`  
**Lines:** 53-108 (`TaskSoundServer` class)

```python
class TaskSoundServer(threading.Thread):
    def __init__(self):
        # Bind UDP socket to localhost:6000
        HOST = "127.0.0.1"
        PORT = 6000
        self.s = socket.socket(type=socket.SOCK_DGRAM)
        self.s.bind((HOST, PORT))
        self.flag = False
        self.data = 0
    
    def run(self):
        while True:
            # Block until UDP packet arrives
            self.data = self.s.recv(1024).decode()
            if self.data != 0:
                self.flag = True  # Signal new data available
```

**Client side in** `ConverterKxmlToJson.py`  
**Lines:** 22-28 (socket setup) and 36-39 (sending)

```python
# Socket setup in Program.__init__()
self.port = 6000
self.host = '127.0.0.1'
self.s = socket.socket(type=socket.SOCK_DGRAM)

# Sending metadata when user clicks Start
def StartSystem(self):
    # Format: "counter,woodnumber,processtype,date"
    self.s.sendto(
        (str(self.conversion_counter) + "," + 
         str(woodnumber) + "," + 
         str(proces_type) + "," + 
         str(date_today)).encode(),
        (self.host, self.port)
    )
```

**Receiving in main loop:**  
**Lines:** 535-545

```python
gui_info = server.reedgui()  # Check for new UDP message
if str(gui_info) != "0":
    gui_info_list = gui_info.split(',')
    counter = int(gui_info_list[0])   # Sample counter
    wood = gui_info_list[1]           # Wood number
    process = gui_info_list[2]        # Process type (A, B, C...)
    today = gui_info_list[3]          # Date string
```

**What happens:** The GUI sends a comma-separated string via UDP when the user clicks "Start". This allows the user to set session parameters (wood number, process type, date, starting counter) that are used for naming output files. The data collection script receives this asynchronously without interrupting the recording loop.

---

### 4. Audio Recording (Microphone via PyAudio)

**File:** `task_and_acustics_data_collection.py`  
**Lines:** 370-436 (`SoundRecorderThread` class)

```python
class SoundRecorderThread(threading.Thread):
    def __init__(self):
        self.frames = []
        self.should_stop = False
        self.CHUNK = 1024          # Buffer size
        self.FORMAT = pyaudio.paInt16  # 16-bit audio
        self.CHANNELS = 1          # Mono
        self.RATE = 44100          # 44.1 kHz sample rate
    
    def run(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=self.FORMAT, 
            channels=self.CHANNELS,
            rate=self.RATE, 
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        while not self.should_stop:
            data = stream.read(self.CHUNK)
            self.frames.append(data)  # Accumulate audio chunks
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
```

**Starting/stopping recording in main loop:**  
**Lines:** 555-560 (start) and 575-600 (stop)

```python
# When PLC signal goes HIGH:
if result and not flag:
    flag = True
    recorder = SoundRecorderThread()
    recorder.start()  # Begin recording in background

# When PLC signal goes LOW:
if not result:
    recorder.stop_recording()
    frames = recorder.get_frames()
    save_recording(frames, filename + ".wav")
```

**What happens:** When the PLC trigger goes HIGH, a new `SoundRecorderThread` is created and started. It opens the default microphone and continuously reads audio chunks into memory. When the PLC trigger goes LOW, the thread is stopped and all accumulated frames are written to a WAV file.

---

### 5. KXML File Processing (Screwdriver Data)

**File:** `ConverterKxmlToJson.py` → `folderManager.py`  
**Lines:** 57-69 in ConverterKxmlToJson.py (`SystemLoop`)

```python
def SystemLoop(self):
    folder = wsk3_kxml_detect_folder_entry_field.get()  # "WSK3"
    name = self.FolderHandler.Wsk3FolderDetect(folder)  # Check for .KXML files
    
    if str(name) != "-1":  # File found
        self.conversion_counter += 1
        json = self.FolderHandler.ConvertKxmlToJson(name)  # Parse KXML to JSON
        
        # Save files with proper naming
        self.FolderHandler.SaveKxmlFile(self.conversion_counter, proces_type, woodnumber, name)
        self.FolderHandler.SaveJsonFile(self.conversion_counter, proces_type, woodnumber, json)
```

**File detection in** `folderManager.py`  
**Lines:** 111-123 (`Wsk3FolderDetect`)

```python
def Wsk3FolderDetect(self, foldername):
    for filename in os.listdir(self.path_of_the_directory + '\\' + foldername):
        if filename.endswith('.KXML'):
            name = os.path.join(foldername, filename)
            return name
    return -1  # No KXML file found
```

**KXML to JSON conversion:**  
**Lines:** 126-135 (`ConvertKxmlToJson`)

```python
def ConvertKxmlToJson(self, name):
    time.sleep(0.1)  # Wait for file to be fully written
    with open(name.replace('\\','/')) as xml_file:
        data_dict = xmltodict.parse(xml_file.read())
        return json.dumps(data_dict)
```

**What happens:** The WSK3 software (from Weber) monitors the screwdriver's serial connection and writes a `.KXML` file to the WSK3 folder after each screw-driving cycle. The GUI's `SystemLoop` polls this folder every ~1ms. When a new KXML file appears, it's parsed using `xmltodict`, converted to JSON, and saved to the data folders. The original KXML is moved to `data_kxml/` for archival.

---

### Connection Flow Summary

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        CONNECTION INITIALIZATION                           │
└────────────────────────────────────────────────────────────────────────────┘

  STARTUP SEQUENCE:
  
  1. task_and_acustics_data_collection.py starts
     │
     ├──► snap7.client.Client().connect('172.20.1.148', 0, 1)  [PLC]
     │
     ├──► ModbusReader('172.20.1.50', 502).start()             [Robot]
     │
     └──► TaskSoundServer().start()                            [UDP Server]
           └──► socket.bind(('127.0.0.1', 6000))
  
  2. ConverterKxmlToJson.py starts (GUI)
     │
     ├──► socket(SOCK_DGRAM)                                   [UDP Client]
     │
     └──► FolderManager(os.getcwd())                           [File Handler]

  RUNTIME CONNECTIONS:
  
  ┌─────────────┐ UDP (when Start clicked) ┌─────────────────────────┐
  │     GUI     │ ─────────────────────────►│ TaskSoundServer.run()  │
  │ sendto()    │   "1,wood1,A,05032026"    │ recv() → parse → store │
  └─────────────┘                           └─────────────────────────┘

  ┌─────────────┐ Snap7 (continuous poll)  ┌─────────────────────────┐
  │  Main Loop  │ ─────────────────────────►│   Siemens PLC DB19     │
  │ PLCsignal() │      db_read(19,0,1)      │   Bit 0 = trigger      │
  └─────────────┘                           └─────────────────────────┘

  ┌─────────────┐ Modbus TCP (background)  ┌─────────────────────────┐
  │ModbusReader │ ─────────────────────────►│   UR10 Controller      │
  │   .run()    │ read_holding_registers()  │   Registers 400-450    │
  └─────────────┘                           └─────────────────────────┘

  ┌─────────────┐ PyAudio (during cycle)   ┌─────────────────────────┐
  │SoundRecorder│ ─────────────────────────►│     Microphone         │
  │   .run()    │   stream.read(1024)       │   44.1kHz PCM data     │
  └─────────────┘                           └─────────────────────────┘

  ┌─────────────┐ File System (polling)    ┌─────────────────────────┐
  │ SystemLoop  │ ─────────────────────────►│   WSK3 Folder          │
  │Wsk3Detect() │   os.listdir('WSK3')      │   *.KXML files         │
  └─────────────┘                           └─────────────────────────┘
```

---

## Quick Start

### Prerequisites
1. Open **WSK3** software from Weber and connect the C30S controller via USB
2. Configure WSK3 to save KXML files to the `WSK3` folder in this project
3. Ensure network connectivity to:
   - UR10 Robot: `172.20.1.50:502`
   - Siemens PLC: `172.20.1.148`

### Running the System

```batch
# Install dependencies (first time only)
install_dependencies.bat

# Launch the system
run.bat
```

This starts both scripts:
- `task_and_acustics_data_collection.py` (background data collection)
- `ConverterKxmlToJson.py` (GUI application)

### GUI Usage
1. Enter **Wood Number** and **Process Type** (e.g., A, B, C)
2. Verify the **Date** field
3. Click **Start** to begin monitoring
4. The system will automatically:
   - Detect PLC trigger signals
   - Record robot data and audio during each screw-driving cycle
   - Convert KXML files to JSON when they appear
   - Save all data with consistent naming

---

## Network Configuration

```
┌─────────────────────────────────────────────────────────────────┐
│                    NETWORK TOPOLOGY                             │
└─────────────────────────────────────────────────────────────────┘

              Industrial Network (172.20.1.x)
    ┌─────────────────────────────────────────────────┐
    │                                                 │
    │    ┌──────────┐    ┌──────────┐    ┌────────┐  │
    │    │  UR10    │    │  PLC     │    │   PC   │  │
    │    │  .50     │    │  .148    │    │  .xxx  │  │
    │    └────┬─────┘    └────┬─────┘    └───┬────┘  │
    │         │               │              │       │
    │    ─────┴───────────────┴──────────────┴─────  │
    │                                                 │
    └─────────────────────────────────────────────────┘

              Local PC Connections
    ┌─────────────────────────────────────────────────┐
    │                                                 │
    │    ┌──────────────┐     ┌──────────────────┐   │
    │    │ Screwdriver  │ USB │       PC         │   │
    │    │ C30S         │────►│                  │   │
    │    └──────────────┘     │  ┌────────────┐  │   │
    │                         │  │ Microphone │  │   │
    │    ┌──────────────┐     │  │ (Audio In) │  │   │
    │    │ WSK3 Software│     │  └────────────┘  │   │
    │    │ (Monitors    │     │                  │   │
    │    │  serial port)│     └──────────────────┘   │
    │    └──────────────┘                            │
    │                                                 │
    └─────────────────────────────────────────────────┘
```

---

## File Naming Convention

All output files follow this pattern:
```
{date}{wood}{process}{counter}.{extension}
```

**Example:** `020320261A5.csv`
- `02032026` = Date (March 2, 2026)
- `1` = Wood number
- `A` = Process type
- `5` = Sample counter
- `.csv` = Robot kinematics data

---

## Dependencies

```
pandas          # Data manipulation
numpy           # Numeric operations
pyModbusTCP     # Robot communication
snap7           # PLC communication
pyaudio         # Audio recording
wave            # WAV file handling
xmltodict       # KXML parsing
tkinter         # GUI framework
```

Install via: `pip install -r requirements.txt` or run `install_dependencies.bat`

