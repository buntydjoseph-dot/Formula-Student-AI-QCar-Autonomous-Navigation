
# Formula Student AI – QCar Autonomous Navigation

AI-driven autonomous navigation system developed for **Formula Student AI**
using the **Quanser QCar** platform and **QLabs** simulation environment.

This project focuses on perception, localisation, and control pipelines
required for autonomous lap navigation in a Formula Student–style cone-based
racing environment.

---

## Project Overview

The objective of this project is to design and evaluate an autonomous
navigation system capable of completing Formula Student AI tracks using
vision-based perception, state estimation, and real-time control algorithms.

The system was developed and tested using Quanser QCar hardware and QLabs
simulation, following constraints and scenarios inspired by Formula Student
AI competitions.

---

## System Architecture

The autonomous stack is structured as:

**Perception → Localisation → Control → Vehicle Actuation**

- **Perception**
  - Vision-based cone detection using a YOLO-based pipeline
  - Dataset capture, annotation, and training workflows

- **Localisation**
  - Extended Kalman Filter (EKF) for state estimation
  - Sensor fusion of vehicle motion and perception outputs

- **Control**
  - Baseline, advanced, and experimental navigation strategies
  - Path tracking and steering control suitable for FS-AI tracks

- **Simulation**
  - Track generation and environment setup in QLabs
  - Formula Student–style layouts for testing and validation

---

## Project Overview

The objective of this project is to design and evaluate an autonomous
navigation system capable of completing Formula Student AI tracks using
vision-based perception, state estimation, and real-time control algorithms.

The system was developed and tested using Quanser QCar hardware and QLabs
simulation, following constraints and scenarios inspired by Formula Student
AI competitions.

---

## System Architecture

The autonomous stack is structured as:

**Perception → Localisation → Control → Vehicle Actuation**

- **Perception**
  - Vision-based cone detection using a YOLO-based pipeline
  - Dataset capture, annotation, and training workflows

- **Localisation**
  - Extended Kalman Filter (EKF) for state estimation
  - Sensor fusion of vehicle motion and perception outputs

- **Control**
  - Baseline, advanced, and experimental navigation strategies
  - Path tracking and steering control suitable for FS-AI tracks

- **Simulation**
  - Track generation and environment setup in QLabs
  - Formula Student–style layouts for testing and validation

---

## Key Technologies

- Python
- Quanser QCar & QLabs
- YOLO-based computer vision
- Extended Kalman Filter (EKF)
- Path tracking and control algorithms
- Formula Student AI–style simulation environments

---

## Notes

- Trained model weights and large datasets are intentionally excluded.
- Users are expected to generate or provide their own datasets and models.
- No proprietary Quanser source code or licensed assets are included.

---

## Disclaimer

This project was developed for **academic and research purposes**.
It does not contain proprietary, confidential, or licensed source code from
Quanser or any third-party organisations.
---

## Prerequisites

To run or extend this project, users must ensure the following are installed
and configured correctly **before** proceeding:

- **Quanser QCar libraries and dependencies** installed from the **official
  Quanser repositories** only.
- A valid installation of **Quanser QLabs**, properly configured on the host
  system.
- Appropriate licenses and permissions required by Quanser software.

This repository does **not** include any Quanser proprietary libraries or
binaries. Users are responsible for installing and configuring all required
Quanser software independently.

Refer to the official Quanser documentation for installation and setup
instructions.
