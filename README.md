# Screwing Cell Task and Acoustics Data Collection

This system collects synchronized data from a screwing-cell test rig, capturing robot kinematics, screwdriver torque/angle data, and acoustic signals during screw-driving operations. The collected data is then cleaned, visualized, and processed through feature engineering and machine learning for classifying screw-driving operations as **Normal** or **Under** (under-torqued / faulty).

---

## Table of Contents

- [Project Overview](#project-overview)
- [Data Pipeline](#data-pipeline)
- [Data Collection](#data-collection)
- [Data Formats](#data-formats)
- [Data Cleaning](#data-cleaning)
- [Data Analysis & Visualization](#data-analysis--visualization)
- [Feature Engineering](#feature-engineering)
- [Machine Learning](#machine-learning)
- [Folder Structure](#folder-structure)
- [Scripts Reference](#scripts-reference)
- [System Communication Overview](#system-communication-overview)

---

## Project Overview

This project implements a complete pipeline for data-driven quality monitoring of an industrial screwing cell:

1. **Data Collection** — Synchronized capture of robot kinematics, screwdriver sensor data, and audio from a UR10 robot / Weber C30S screwdriver / Siemens S7 PLC setup.
2. **KXML-to-JSON Conversion** — Raw screwdriver data files (KXML/XML) are converted to JSON via a Tkinter GUI.
3. **Data Cleaning** — Time normalization, negative-value clipping, NaN handling, and encoding fixes.
4. **Visualization** — Overview and per-file plots of all sensor channels.
5. **Feature Engineering** — Automated time-series feature extraction using tsfresh.
6. **Machine Learning** — Classification of screwing operations into Normal vs. Under categories.

---

## Data Pipeline

```
  WSK3 (KXML files)    UR10 Robot (Modbus)    Microphone (PyAudio)    PLC Trigger (Snap7)
        │                      │                      │                       │
        ▼                      ▼                      ▼                       │
  ConverterKxmlToJson.py   task_and_acustics_data_collection.py  ◄────────────┘
        │                      │
        ▼                      ▼
   data/ (JSON+KXML)      data/ (CSV+WAV)
        │                      │
        └──────────┬───────────┘
                   ▼
           data_opsamling/          ← manually organized into Normal/ and Under/
           (Normal/ + Under/)
                   │
                   ▼
           data_cleaning.py
                   │
                   ▼
           data_opsamling_cleaned/
           (Normal/ + Under/)
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
  visualize_020320261.py   Feature_engineering/code.py
                                    │
                                    ▼
                           features_extracted.csv
                           features_selected.csv
                                    │
                                    ▼
                           ML_models_training/code.py
```

---

## Data Collection

Two applications run simultaneously during experiments:

### task_and_acustics_data_collection.py (Background Service)

Synchronizes three data streams, triggered by the PLC signal:

| Stream | Source | Protocol | Rate | Output |
|--------|--------|----------|------|--------|
| Robot kinematics | UR10 via Modbus TCP (172.20.1.50:502) | Modbus | ~400 Hz | `.csv` |
| Audio | Microphone via PyAudio | PCM 16-bit | 44.1 kHz | `.wav` |
| PLC trigger | Siemens S7 PLC (172.20.1.148) | Snap7 | Polling | Start/Stop signal |

When the PLC trigger goes **HIGH**, the script starts recording robot registers and audio. When it goes **LOW**, both recordings are saved to disk.

### ConverterKxmlToJson.py (GUI Application)

A Tkinter-based GUI that:
- Monitors the `WSK3/` folder for new `.KXML` files from the Weber screwdriver controller
- Converts KXML → JSON using `xmltodict` (via `folderManager.py`)
- Sends session metadata (wood number, process type, date, counter) to the data collection script over **UDP port 6000**
- Allows relabeling and deleting samples

### Naming Convention

Files follow the pattern: `{date}{wood_number}{process_type}{screw_number}`

Example: `120320261A3.json` = Date 12/03/2026, Wood #1, Process A (Normal), Screw #3

- **Process types**: `A` = Normal, `B/C/...` = various fault conditions

---

## Data Formats

### CSV Files (Robot Kinematics)

Each CSV file represents one screw-driving cycle recorded from the UR10 robot:

| Column | Unit | Description |
|--------|------|-------------|
| `Time (ms)` | milliseconds | Timestamp (sampled at ~2-4 ms intervals, irregular) |
| `TCP_x (mm)` | mm | Tool Center Point X position |
| `TCP_y (mm)` | mm | Tool Center Point Y position |
| `TCP_z (mm)` | mm | Tool Center Point Z position |
| `TCP_rx (mm)` | radians | Tool orientation rotation X |
| `TCP_ry (mm)` | radians | Tool orientation rotation Y |
| `TCP_rz (mm)` | radians | Tool orientation rotation Z |
| `Robot_I (A)` | amperes | Robot joint current |

**Example** (first rows of a CSV):
```
Time (ms),TCP_x (mm),TCP_y (mm),TCP_z (mm),TCP_rx (mm),TCP_ry (mm),TCP_rz (mm),Robot_I (A)
0.0,-80.9,-827.9,177.5,1.234,1.181,-1.205,1.232
4.005,-80.9,-827.9,177.5,1.234,1.181,-1.205,1.232
6.005,-80.9,-827.9,177.6,1.234,1.181,-1.205,1.197
```

**Characteristics:**
- ~250-500 rows per file (varies per cycle duration)
- Time intervals are non-uniform (typically 2-4 ms gaps)
- Some files may have a non-zero starting time (fixed during cleaning)

### JSON Files (Screwdriver Data)

Each JSON file contains data from one screw-driving cycle captured by the Weber C30S controller. The data is stored in a nested XML-like structure:

```
JSON root
└── XML_Data
    └── Wsk3Vectors
        ├── X_Axis          → time axis (ms)
        └── Y_AxesList
            └── AxisData[]  → one entry per signal channel
                ├── Header {Name, Unit}
                └── Values {float[]}
```

| Signal | Unit | Description |
|--------|------|-------------|
| `Nset` | rpm | Set speed (spindle target RPM) |
| `Torque` | Nm | Measured torque |
| `Current` | A | Motor current |
| `Angle` | ° | Rotation angle |
| `Depth` | mm | Screw penetration depth |

**Characteristics:**
- Uniform time steps (fixed sampling interval)
- Typically ~200-400 data points per channel
- Angle unit may have encoding artifacts (`Â°` instead of `°`)
- Torque and Current may contain small negative values (sensor noise)

---

## Data Cleaning

### data_cleaning.py

Cleans raw data from `data_opsamling/` and writes cleaned files to `data_opsamling_cleaned/`.

**Usage:**
```bash
python data_cleaning.py Normal        # clean one subfolder
python data_cleaning.py Under         # clean another subfolder
python data_cleaning.py --all         # clean all subfolders
```

**Cleaning operations performed:**

| # | Data Type | Operation | Reason |
|---|-----------|-----------|--------|
| 1 | CSV | Shift time to start at 0 ms | Some files have non-zero start time |
| 2 | CSV | Report and drop rows with NaN | Data integrity |
| 3 | JSON | Clip negative Torque values to 0 | Sensor noise produces small negative readings |
| 4 | JSON | Clip negative Current values to 0 | Sensor noise produces small negative readings |
| 5 | JSON | Replace NaN values with 0 | Data integrity |
| 6 | JSON | Fix Angle unit encoding (`Â°` → `°`) | UTF-8 encoding artifact from KXML conversion |

### Dataset Organization (data_opsamling/)

Before cleaning, the raw paired data files are organized into labeled subfolders:

| Subfolder | Label | Description | Files |
|-----------|-------|-------------|-------|
| `Normal/` | 0 | Normal screw-driving operations | 5 paired CSV+JSON samples |
| `Under/` | 1 | Under-torqued / faulty operations | 3 paired CSV+JSON samples |

Each sample consists of a matching `.csv` (robot data) and `.json` (screwdriver data) file with the same filename stem.

### cleaning_examples.py

A diagnostic script that demonstrates each cleaning step with concrete examples from the dataset. It reports:
1. **Time normalization** — which files have non-zero start times
2. **Resampling analysis** — time gap statistics (min/max/mean/std)
3. **Idle/startup phase detection** — how many rows are flat before movement begins
4. **Outlier analysis** — IQR-based outlier counts in Robot_I and Torque
5. **Screwing segmentation** — ramp-up / active / ramp-down phases in Nset
6. **Encoding issues** — raw unit strings for Angle
7. **Length normalization** — row count ranges across all files

### analyze_data_quality.py

Produces a full data quality report for all subfolders in `data_opsamling/`, including:
- Null/duplicate/infinity checks for CSV and JSON
- Time monotonicity verification
- Constant column detection
- IQR-based outlier detection (3x IQR)
- Flat-region detection (>50% zero-diff)
- Cross-file row count consistency
- Negative value detection across all channels

### compare_csv_json_timing.py

Compares the timing characteristics between paired CSV and JSON files:
- Row counts, mean time step, and total duration for both file types
- Checks whether durations match within 100 ms tolerance
- Highlights that CSV has irregular time steps vs. JSON's uniform sampling

---

## Data Analysis & Visualization

### visualize_020320261.py

Generates plots from `data_opsamling/` into `visualizations/`:

- **csv_overview.png** — All CSV signals (TCP positions, orientations, current) overlaid by subfolder, color-coded by label
- **json_overview.png** — All JSON signals (Nset, Torque, Current, Angle, Depth) overlaid by subfolder
- **Per-file detail plots** — Individual detailed views of each sample showing all CSV and JSON channels

### Extract_data_from_csv_and_json/exstract_data.py

Simple utility to load and inspect CSV and JSON files from a data folder. Prints column names and first rows for quick exploration.

---

## Feature Engineering

### Feature_engineering/code.py

Extracts time-series features from the **cleaned** data (`data_opsamling_cleaned/`) using the [tsfresh](https://tsfresh.readthedocs.io/) library.

**Usage:**
```bash
python Feature_engineering/code.py               # extract + select features
python Feature_engineering/code.py --no-select    # extract only, skip selection
```

**Process:**
1. Pairs CSV and JSON files by matching filename stems in `Normal/` and `Under/`
2. Loads each pair into tsfresh long format with a shared `sample_id`
3. Extracts features from **robot data** (CSV) and **screwdriver data** (JSON) separately using `EfficientFCParameters`
4. Imputes missing values
5. Prefixes columns (`robot_` / `screw_`) and merges into one feature matrix
6. Optionally runs `tsfresh.select_features` to keep only statistically relevant features

**Outputs:**

| File | Description |
|------|-------------|
| `features_extracted.csv` | All tsfresh features from both data sources |
| `features_selected.csv` | Subset of relevant features (after statistical selection) |
| `labels.csv` | Class labels (0 = Normal, 1 = Under) |

---

## Machine Learning

### ML_models_training/code.py

Placeholder for classification model training. Intended to consume the feature matrices produced by the feature engineering step to train classifiers (e.g., Random Forest, SVM, etc.) to distinguish Normal from Under screwing operations.

---

## Folder Structure

```
VT2/
├── data/                          Raw collected data (per experiment session)
│   ├── 020320261/                 Date+wood session folders
│   │   ├── 020320261A1.csv        Robot kinematics
│   │   ├── 020320261A1.json       Screwdriver data (converted from KXML)
│   │   └── ...
│   └── ...
├── data_kxml/                     Archived original KXML files
├── data_opsamling/                Labeled dataset for ML (manually organized)
│   ├── Normal/                    Normal screwing samples (CSV+JSON pairs)
│   └── Under/                     Under-torqued samples (CSV+JSON pairs)
├── data_opsamling_cleaned/        Cleaned version of data_opsamling/
│   ├── Normal/
│   └── Under/
├── visualizations/                Generated plots from visualize_020320261.py
├── visualizations_cleaned/        Generated plots from cleaned data
├── Feature_engineering/
│   ├── code.py                    tsfresh feature extraction
│   ├── features_extracted.csv     Output: all features
│   ├── features_selected.csv      Output: selected features
│   └── labels.csv                 Output: class labels
├── ML_models_training/
│   └── code.py                    Model training (placeholder)
├── Extract_data_from_csv_and_json/
│   └── exstract_data.py           Data loading utility
├── WSK3/                          Weber screwdriver KXML drop folder
│
├── task_and_acustics_data_collection.py   Main data collection service
├── ConverterKxmlToJson.py                 GUI + KXML converter
├── folderManager.py                       File system utility
├── data_cleaning.py                       Data cleaning pipeline
├── cleaning_examples.py                   Cleaning step demonstrations
├── analyze_data_quality.py                Data quality reporting
├── compare_csv_json_timing.py             CSV vs JSON timing analysis
├── visualize_020320261.py                 Visualization generator
├── servertest.py                          UDP server timing test
├── Test.py                                Hello world test
├── _test_tsfresh.py                       tsfresh minimal smoke test
├── install_dependencies.bat               Dependency installer
└── README.md                              This file
```

---

## Scripts Reference

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `task_and_acustics_data_collection.py` | Collect robot + audio data | UR10 Modbus, PLC, Microphone | `data/` CSV + WAV |
| `ConverterKxmlToJson.py` | GUI, KXML→JSON conversion | WSK3 folder, user input | `data/` JSON, `data_kxml/` KXML |
| `folderManager.py` | File operations helper | Called by ConverterKxmlToJson | Folder/file management |
| `data_cleaning.py` | Clean CSV+JSON data | `data_opsamling/` | `data_opsamling_cleaned/` |
| `cleaning_examples.py` | Show cleaning step examples | `data_opsamling/` | Console output |
| `analyze_data_quality.py` | Data quality report | `data_opsamling/` | Console output |
| `compare_csv_json_timing.py` | Compare CSV/JSON timing | `data_opsamling/` | Console output |
| `visualize_020320261.py` | Generate plots | `data_opsamling/` | `visualizations/` PNG files |
| `Feature_engineering/code.py` | Extract tsfresh features | `data_opsamling_cleaned/` | Feature CSV files |
| `ML_models_training/code.py` | Train classifiers | Feature CSV files | (placeholder) |
| `Extract_data_from_csv_and_json/exstract_data.py` | Quick data inspection | `data/020320261/` | Console output |
| `servertest.py` | UDP server timing benchmark | — | Timing plot |
| `_test_tsfresh.py` | Smoke test for tsfresh | — | `_test_output.csv` |

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

