import math

import torch
from torch.utils.data import IterableDataset


class SyntheticEpisodeDataset(IterableDataset):
    """Placeholder episode generator for smoke-testing the TNP-D architecture
    end-to-end before real WOFOST-simulated episodes exist.

    Each episode: one location (random lat/lon), a handful of modalities
    observed at random times over a synthetic season, values drawn from a
    per-episode random sinusoid (a different phase/amplitude per modality) so
    the model has a learnable, but entirely synthetic, weather-like signal to
    fit. Context/target set sizes are randomized once per yielded batch, not
    per example, matching how the original TNP-pytorch training loop samples
    batches. Swap this out for the real WOFOST-driven episode dataset once the
    simulation pipeline (see wofost_synthetic_pretraining_plan) lands.

    Each `__iter__` step already yields a full batch (dict of `[B, N, ...]`
    tensors matching `src.models.components.tnpd.Batch`'s fields), so the
    corresponding `DataLoader` should be created with `batch_size=None`.
    """

    def __init__(
        self,
        num_batches: int,
        batch_size: int,
        num_modalities: int,
        min_context: int,
        max_context: int,
        min_target: int,
        max_target: int,
        season_length_days: float = 180.0,
        noise_std: float = 0.05,
        seed: int = 0,
    ):
        super().__init__()
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.num_modalities = num_modalities
        self.min_context = min_context
        self.max_context = max_context
        self.min_target = min_target
        self.max_target = max_target
        self.season_length_days = season_length_days
        self.noise_std = noise_std
        self.seed = seed

    def __len__(self) -> int:
        return self.num_batches

    def _sample_episode_batch(self, generator: torch.Generator) -> dict:
        num_ctx = torch.randint(self.min_context, self.max_context + 1, (1,), generator=generator).item()
        num_tar = torch.randint(self.min_target, self.max_target + 1, (1,), generator=generator).item()
        num_all = num_ctx + num_tar

        lat = torch.empty(self.batch_size, 1).uniform_(35.0, 60.0, generator=generator)
        lon = torch.empty(self.batch_size, 1).uniform_(-10.0, 30.0, generator=generator)
        time = torch.empty(self.batch_size, num_all).uniform_(0.0, self.season_length_days, generator=generator)
        depth = torch.zeros(self.batch_size, num_all)
        modality = torch.randint(0, self.num_modalities, (self.batch_size, num_all), generator=generator)

        x = torch.stack([lat.expand(-1, num_all), lon.expand(-1, num_all), time, depth], dim=-1)

        phase = torch.empty(self.batch_size, self.num_modalities).uniform_(0, 2 * math.pi, generator=generator)
        amplitude = torch.empty(self.batch_size, self.num_modalities).uniform_(0.5, 1.5, generator=generator)
        phase_per_token = torch.gather(phase, 1, modality)
        amplitude_per_token = torch.gather(amplitude, 1, modality)

        period = self.season_length_days / 2
        signal = amplitude_per_token * torch.sin(2 * math.pi * time / period + phase_per_token)
        noise = torch.randn(self.batch_size, num_all, generator=generator) * self.noise_std
        y = (signal + noise).unsqueeze(-1)

        return {
            "xc": x[:, :num_ctx],
            "yc": y[:, :num_ctx],
            "mc": modality[:, :num_ctx],
            "xt": x[:, num_ctx:],
            "yt": y[:, num_ctx:],
            "mt": modality[:, num_ctx:],
        }

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        for _ in range(self.num_batches):
            yield self._sample_episode_batch(generator)
