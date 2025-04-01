import subprocess
import csv


class EnergibridgeLogger:
    def __init__(self, rapl_power_path="vision/metrics/EnergiBridge/target/release/energibridge", output_file="energy_log.csv"):
        self.rapl_power_path = rapl_power_path
        self.output_file = output_file
        self.logger_process = None

    def start_logging(self,duration=20):
        try:
            self.logger_process = subprocess.Popen(
                [self.rapl_power_path, "-o", self.output_file,"--summary","sleep",str(duration)]
            )
            print(f"✅ Energibridge started (output: {self.output_file})")
        except Exception as e:
            print(f"Failed to start Energibridge: {e}")

    def stop_logging(self):
        if self.logger_process:
            self.logger_process.terminate()
            self.logger_process.wait()
            print(" Energibridge logging stopped.")

    def parse_energy(self):
        energy = 0.0
        try:
            with open(self.output_file, "r") as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                if "Energy" in rows[0]:
                    energy = sum(float(row["Energy"]) for row in rows)
                elif "SYSTEM_POWER (Watts)" in rows[0]:
                    power_vals = [float(row["SYSTEM_POWER (Watts)"]) for row in rows]
                    avg_power = sum(power_vals) / len(power_vals)
                    duration_sec = len(power_vals)  # assuming 1 reading/sec
                    energy = avg_power * duration_sec / 3600.0  # convert to Wh
                else:
                    print("Unknown energy format.")
        except Exception as e:
            print(f"Failed to parse Energibridge energy: {e}")
        return energy  # returns in Wh
