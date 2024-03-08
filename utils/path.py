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

import os

# Determine the directory two levels up from the current file's location
CODE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(CODE_PATH, "train")

ENSEMBLE_PATH = os.path.join(CODE_PATH, "ensemble")

ALIGNMENT_PATH = os.path.join(CODE_PATH, "alignment")

DATA_PATH = os.path.join(CODE_PATH, "data")


