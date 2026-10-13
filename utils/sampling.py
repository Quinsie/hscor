"""Batch-256 sampling with the original four-stream occurrence order."""

import math

import torch


class TrainingBatches:
    def __init__(self, population, seed, epoch):
        if population < 1:
            raise ValueError("empty training population")
        generator = torch.Generator().manual_seed(seed + epoch)
        values = torch.randperm(population, generator=generator).tolist()
        padding = math.ceil(population / 4) * 4 - population
        values += (values * math.ceil(padding / population))[:padding]
        self.streams = [values[rank::4] for rank in range(4)]
        self.epoch = epoch

    def __len__(self):
        return math.ceil(len(self.streams[0]) / 64)

    def __iter__(self):
        for start in range(0, len(self.streams[0]), 64):
            yield [
                (index, self.epoch, rank, start + offset)
                for rank, stream in enumerate(self.streams)
                for offset, index in enumerate(stream[start : start + 64])
            ]
