import subprocess
import time
import csv

from codecarbon import EmissionsTracker
from energibridge_logger import EnergibridgeLogger

def track_energy(run_function, *args, **kwargs):
    """
        Wraps the given function call with energy measurement.

        Uses CodeCarbon's EmissionsTracker and EnergibridgeTracker to measure energy usage.

        Args:
            run_function (callable): Function to run (e.g., an inference loop or evaluation routine).
            *args, **kwargs: Arguments to pass to run_function.

        Returns:
            A tuple (function_result, total_energy_wh) where total_energy_wh is the combined energy (Wh).
        """
    cc_tracker = EmissionsTracker(measure_power_secs=20)
    cc_tracker.start()

    eb_logger=EnergibridgeLogger(rapl_power_path="./path/to/rapl-power", output_file="energy_log.csv")
    eb_logger.logger_process = subprocess.Popen(
        [eb_logger.rapl_power_path, "--csv", "--output", eb_logger.output_file]
    )
    print(f"Started Energibridge logging (output: {eb_logger.output_file})")

    result = run_function(*args, **kwargs)
    eb_logger.stop_logging()
    cc_energy_wh = cc_tracker.stop()
    try:
        total_energy=0.0
        with open(eb_logger.output_file, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                total_energy+=float(row["Energy"])
    except Exception as e:
        print(f"An error occurred while parsing the energy log file: {e}")
        total_energy=0.0
    total_energy_wh =(cc_energy_wh+total_energy)/2.0
    return result, total_energy_wh
if __name__=="__main__":
    def dummy_eval():
        time.sleep(10)
        return (0.85,100)
    result, energy_wh = track_energy(dummy_eval)
    print(f"Dummy evaluation function result: {result}")
    print(f"Total energy measured: {energy_wh:.6} Wh")
