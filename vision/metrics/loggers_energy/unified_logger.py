from vision.metrics.loggers_energy.codecarbon_tracker import CarbonTracker
from vision.metrics.loggers_energy.energibridge_logger import EnergibridgeLogger


class UnifiedLogger:
    def __init__(self, experiment_name='experiment', cc_output_dir='emissions', eb_output_file='energy_log.csv', rapl_power_path='vision/metrics/EnergiBridge/target/release/rapl-power'):
        self.cc_tracker = CarbonTracker(name=experiment_name, output_dir=cc_output_dir)
        self.eb_logger = EnergibridgeLogger(rapl_power_path=rapl_power_path, output_file=eb_output_file)

    def start(self):
        print("Starting Unified Logger...")
        self.cc_tracker.start()
        self.eb_logger.start_logging()

    def stop(self):
        cc_emissions = self.cc_tracker.stop()  # kWh
        eb_energy_wh = self.eb_logger.parse_energy()  # Wh
        cc_energy_wh = cc_emissions * 1000  # Convert kWh to Wh

        total_energy_wh = (cc_energy_wh + eb_energy_wh) / 2.0

        return {
            "codecarbon_energy_wh": cc_energy_wh,
            "energibridge_energy_wh": eb_energy_wh,
            "total_energy_wh": total_energy_wh
        }