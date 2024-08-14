import logging
import os
import random

import numpy
import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Reproducible:
    seed: int = None

    def apply_seed(self):
        """Sets a seed for the run in order to make the results reproducible."""
        logger.info(f"Setting {self.seed=}")
        os.environ["PYTHONHASHSEED"] = str(self.seed)

        np.random.seed(self.seed)

        # whether to use the torch autotuner and find the best algorithm for current hardware
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        torch.cuda.manual_seed(self.seed)
        torch.manual_seed(self.seed)
        numpy.random.seed(self.seed)
        random.seed(self.seed)

    def get_seed(self):
        return self.seed

    def set_seed(self, seed):
        self.seed = seed
        return self

    def seed_worker(self, seed_worker):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    def get_generator(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        return g
