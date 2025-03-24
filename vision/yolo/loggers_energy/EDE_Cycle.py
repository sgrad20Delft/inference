import math


class EDECycleCalculator:
    @staticmethod
    def compute_ede_cycle(mAP, flops, train_energy, inference_energy, alpha):
        total_energy = train_energy + inference_energy
        try:
            denominator = math.log(flops) * total_energy
            if denominator == 0:
                return float('inf')
        except ValueError:
            raise ValueError("FLOPS must be a positive number greater than 0.")
        ede_cycle = (mAP ** alpha) / denominator

        return ede_cycle
