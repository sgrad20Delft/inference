import subprocess
import csv
from pathlib import Path


class EnergibridgeLogger:
    def __init__(self, rapl_power_path="vision/metrics/EnergiBridge/target/release/energibridge", output_file="energy_log.csv"):
        self.rapl_power_path = rapl_power_path
        self.output_file = str(Path(output_file).resolve())
        self.logger_process = None

    def start_logging(self,duration=20):
        try:
            self.logger_process = subprocess.Popen(
                [self.rapl_power_path, "-o", self.output_file,"--summary","sleep",str(duration)]
            )
            print(f"Energibridge started (output: {self.output_file})")
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
        print(f"Estimated Total energy: {energy:.2f} Wh")
        return energy  # returns in Wh

    def parse_gpu_energy(self):
        with open(self.output_file, "r") as f:
            lines = f.readlines()

        if len(lines) < 2:
            return 0.0

        header = lines[0].strip().split(",")
        power_idx = header.index("SYSTEM_POWER (Watts)")
        cpu_usage_indices = [i for i, h in enumerate(header) if h.startswith("CPU_USAGE_")]

        gpu_energy_wh = 0.0
        for i in range(1, len(lines) - 1):
            row1 = list(map(float, lines[i].strip().split(",")))
            row2 = list(map(float, lines[i + 1].strip().split(",")))

            delta_sec = (row2[1] - row1[1]) / 1000.0  # Time in seconds
            system_power = (row1[power_idx] + row2[power_idx]) / 2.0

            avg_cpu_usage = sum(row1[i] for i in cpu_usage_indices) / len(cpu_usage_indices)
            estimated_cpu_power = system_power * (avg_cpu_usage / 100.0)  # crude approximation
            estimated_gpu_power = max(0.0, system_power - estimated_cpu_power)

            gpu_energy_wh += estimated_gpu_power * delta_sec / 3600  # Convert to Wh

        print(f"Estimated GPU energy: {gpu_energy_wh:.2f} Wh")
        return gpu_energy_wh

