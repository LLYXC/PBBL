import io
import torch
import numpy as np


class MultiDimAverageMeter(object):
    def __init__(self, dims):
        self.dims = dims
        self.cum = torch.zeros(np.prod(dims))
        self.cnt = torch.zeros(np.prod(dims))
        self.idx_helper = torch.arange(np.prod(dims), dtype=torch.long).reshape(
            *dims
        )

    def add(self, vals, idxs):
        flattened_idx = torch.stack(
            [self.idx_helper[tuple(idxs[i])] for i in range(idxs.size(0))],
            dim=0,
        )
        self.cum.index_add_(0, flattened_idx, vals.view(-1).float())
        self.cnt.index_add_(
            0, flattened_idx, torch.ones_like(vals.view(-1), dtype=torch.float)
        )
        
    def get_mean(self):
        return (self.cum / self.cnt).reshape(*self.dims)

    def reset(self):
        self.cum.zero_()
        self.cnt.zero_()


import logging
from logging.config import dictConfig

# [%(asctime)s] [%(levelname)s]

def set_logging(logger_name, level, work_dir):
    LOGGING = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "format": f"%(message)s"
            },
        },
        "handlers": {
            "console": {
                "level": f"{level}",
                "class": "logging.StreamHandler",
                'formatter': 'simple',
            },
            'file': {
                'level': f"{level}",
                'formatter': 'simple',
                'class': 'logging.FileHandler',
                'filename': f'{work_dir if work_dir is not None else "."}/train.log',
                'mode': 'a',
            },
        },
        "loggers": {
            "": {
                "level": f"{level}",
                "handlers": ["console", "file"] if work_dir is not None else ["console"],
            },
        },
    }
    dictConfig(LOGGING)
    logging.info(f"Log level set to: {level}")
