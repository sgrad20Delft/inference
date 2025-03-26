from time import time
import os
from codecarbon import EmissionsTracker
from energibridge_logger import EnergibridgeLogger
from EDE_Cycle import EDECycleCalculator
from vision.yolo.loggers_energy.UnifiedEnergyLoggerInterface import UnifiedEnergyLoggerInterface


class UnifiedLogger(UnifiedEnergyLoggerInterface):
    def __init__(self,log_dir="logs",model_name="YOLOv5s",power=30,interval=0.5):
        os.makedirs(log_dir,exist_ok=True)
        self.carbon=EmissionsTracker(output_dir=log_dir)



