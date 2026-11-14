{\rtf1\ansi\ansicpg1252\cocoartf2907
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Occhio \'96 project context\
\
Real-time assistive vision for blind users (Street mode) and worker safety (Industrial mode).\
Hackathon: The Big Hack 2026, Mare Group FieldVision challenge. Deadline Sun 15 Nov 12:00.\
\
## Hardware\
MacBook Air M5, fanless. Use device="mps". Keep models small (yolo11n, Depth-Anything-V2-Small).\
\
## Pipeline\
capture -> detect (YOLO11n) -> track (ByteTrack) -> reason (tracks, spatial, reliability, priority) -> speak + overlay\
\
## Rules for code\
- Python 3.11, type hints, small modules as in the folder structure. No global state except config.\
- Real-time first: never block the main loop. Speech, LLM calls and depth run in background threads.\
- Target >= 20 FPS end to end. Print FPS and latency every second.\
- All user-facing strings come from config/profiles.py (English by default, Italian with --lang it).\
- Every logic module (tracks, spatial, priority, reliability) has pytest tests with fake detections.\
- Keep `python main.py --mode street --source 0` working after every change.\
- Do not add new dependencies without asking.}