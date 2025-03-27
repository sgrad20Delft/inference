import subprocess
import time


class EnergibridgeLogger:
    def __init__(self, rapl_power_path="./path/to/rapl-power", output_file="energy_log.csv"):
        self.rapl_power_path = rapl_power_path
        self.output_file = output_file
        self.logger_process = None

    def start_logging(self, duration=10):
        try:
            # Start the subprocess
            self.logger_process = subprocess.Popen(
                [self.rapl_power_path, "--csv", "--output", self.output_file]
            )
            print(f"Started energy logging (output: {self.output_file})")
            time.sleep(duration)
        except Exception as e:
            print(f"An error occurred while starting the logger: {e}")
        finally:
            self.stop_logging()

    def stop_logging(self):
        if self.logger_process:
            self.logger_process.terminate()
            self.logger_process.wait()
            print("Energy logging terminated.")

if __name__ == "__main__":
    energy_logger = EnergibridgeLogger(rapl_power_path="./path/to/rapl-power", output_file="energy_log.csv")
    energy_logger.start_logging(duration=10)
