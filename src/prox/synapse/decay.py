import time
from dataclasses import dataclass

DECAY_LAMBDA = 0.05
DECAY_THRESHOLD = 0.1


@dataclass
class DecayScheduler:
    lambda_factor: float = DECAY_LAMBDA
    threshold: float = DECAY_THRESHOLD

    def compute_strength(self, trace: dict, current_time: float = None) -> float:
        if current_time is None:
            current_time = time.time()
        metadata = trace.get("metadata", {})
        created_at = metadata.get("created_at", current_time)
        last_accessed = metadata.get("last_accessed", created_at)
        access_count = metadata.get("access_count", 0)

        age = (current_time - created_at) / 3600.0
        idle_time = (current_time - last_accessed) / 3600.0

        decay = pow(2.7182818, -self.lambda_factor * idle_time)
        boost = 1.0 + 0.1 * min(access_count, 10)
        strength = decay * boost

        return max(0.0, strength)

    def should_prune(self, trace: dict) -> bool:
        strength = self.compute_strength(trace)
        return strength < self.threshold

    def prune_candidates(self, traces: list[dict], max_age_days: float = 30.0) -> list[str]:
        current_time = time.time()
        to_prune = []
        for trace in traces:
            metadata = trace.get("metadata", {})
            age_seconds = current_time - metadata.get("created_at", current_time)
            age_days = age_seconds / 86400.0
            if age_days > max_age_days:
                to_prune.append(trace.get("id", ""))
                continue
            if self.should_prune(trace):
                to_prune.append(trace.get("id", ""))
        return to_prune
