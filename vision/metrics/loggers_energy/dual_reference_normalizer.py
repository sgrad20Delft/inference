import json

class DualReferenceNormalizer:
    def __init__(self, reference_config):
        self.reference_config = reference_config
        with open(self.reference_config, 'r') as f:
            self.refs = json.load(f)

    def normalize_energy(self, measured_energy_wh):
        low = self.refs['low_energy_baseline_wh']
        high = self.refs['high_energy_baseline_wh']
        normalized_energy = (measured_energy_wh - low) / (high - low)
        return normalized_energy

    def accuracy_penalty(self, accuracy, task_type):
        threshold = self.refs['accuracy_threshold'][task_type]
        if accuracy >= threshold:
            return 1.0
        return accuracy / threshold  # Penalty factor < 1 if accuracy is lower
