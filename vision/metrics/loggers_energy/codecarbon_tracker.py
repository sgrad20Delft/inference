from codecarbon import EmissionsTracker

class CarbonTracker:
    def __init__(self, name='experiment', output_dir='emissions'):
        self.tracker = EmissionsTracker(project_name=name, output_dir=output_dir,allow_multiple_runs=True)

    def start(self):
        self.tracker.start()

    def stop(self):
        emissions = self.tracker.stop()
        return emissions
