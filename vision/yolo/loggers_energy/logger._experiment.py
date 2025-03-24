from time import time
import os
from codecarbon import EmissionsTracker
from energibridge_logger import EnergibridgeLogger
from EDE_Cycle import EDECycleCalculator

class UnifiedLogger:
    def __init__(self,log_dir="logs",model_name="YOLOv5s",power=30,interval=0.5):
        os.makedirs(log_dir,exist_ok=True)
        self.carbon=EmissionsTracker(project_name=model_name,output_dir=log_dir)
        self.energy=EnergibridgeLogger(power=power,interval=interval)
        self.cycle=EDECycleCalculator(power=power,interval=interval)
        self.start_time=time()


