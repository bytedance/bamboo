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

# constants for approximating erfc function
ewald_f = 1.12837917
ewald_p = 0.3275911
ewald_a = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429]

# constants in SI unit
angstrom = 1.0e-10 
bohr = 5.29177e-11
electron_charge = 1.60217663e-19 
kcal_mol = 6.9477e-21 
debye = 3.33564e-30 
coulomb_constant = 8.9875517923e+9 
atm_pressure = 1.01325e+5
hartree = 4.3597447222071e-18
kcal_mol = 6.9477e-21

# unit conversion used in Bamboo
# debye_ea: 0.20819427381112157
debye_ea = debye / (electron_charge * angstrom) 
hartree_kcal_mol = hartree / kcal_mol
bohr_angstrom = bohr / angstrom

# ele_factor: 332.06349451357806
ele_factor = coulomb_constant * electron_charge * electron_charge / kcal_mol / angstrom

# nktv2p: 68568.46780162843
nktv2p = kcal_mol / angstrom / angstrom / angstrom / atm_pressure


nelems = 87 # placeholder, H to Rn

# Hardcode Li and LI.
atom_mapper = {'H': 1, 'He': 2, 'LI': 3,
               'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
               'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P' : 15, 'S': 16, 'Cl': 17, 'Ar': 18,
               'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 
               'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 
               'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
               'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44,
               'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te':52,
               'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
               'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68,
               'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76,
               'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86}
