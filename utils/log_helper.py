# -----  BAMBOO: Bytedance AI Molecular Booster -----
# Copyright 2022-2024 Bytedance Ltd. and/or its affiliates 

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

import logging
import os
from typing import Optional


def create_logger(name, log_file: Optional[str] = None, screen: bool = True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevents log messages from propagating to parent loggers

    # Clear handles if any.
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s")

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if screen:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
