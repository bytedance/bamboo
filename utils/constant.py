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
electron_charge = 1.60217663e-19 
kcal_mol = 6.9477e-21 
debye = 3.33564e-30 
coulomb_constant = 8.9875517923e+9 
atm_pressure = 1.01325e+5

# unit conversion used in Bamboo
# debye_ea: 0.20819427381112157
debye_ea = debye / (electron_charge * angstrom) 

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

# element parameter C6 
element_c6 = [1.00000000e+00, 6.92199711e+01, 2.14720701e+01, 
              6.24604751e+03, 1.86461658e+03, 7.56226275e+02, 4.40658854e+02, 2.82242132e+02, 1.76224475e+02, 1.14964726e+02, 8.66654253e+01, 
              9.66266029e+03, 5.56711603e+03, 4.49799246e+03, 3.23170576e+03, 2.40493377e+03, 1.79471121e+03, 1.25899491e+03, 8.90770544e+02, 
              2.59943123e+04, 1.68207660e+04, 8.26273588e+03, 7.77095586e+03, 6.69641292e+03, 4.59291831e+03, 
              4.39271718e+03, 3.26533023e+03, 3.02206176e+03, 3.10171893e+03, 3.40489704e+03, 3.24257643e+03, 
              4.38913208e+03, 3.72928104e+03, 3.38141640e+03, 2.93024731e+03, 2.31739828e+03, 1.79682631e+03]
# element parameter r0 
element_r0 = [1.        , 2.18230009, 1.73469996, 
              3.49559999, 3.09820008, 3.21600008, 2.91030002, 2.62249994, 2.48169994, 2.29959989, 2.13739991,
              3.70819998, 3.48390007, 4.01060009, 3.79169989, 3.50169992, 3.31069994, 3.10459995, 2.91479993,
              4.24109983, 4.10349989, 3.89030004, 3.76419997, 3.72110009, 3.44140005, 
              3.54620004, 3.44210005, 3.43269992, 3.34619999, 3.30080009, 3.23090005, 
              3.95790005, 3.86190009, 3.6624999 , 3.52679992, 3.36619997, 3.20959997]
