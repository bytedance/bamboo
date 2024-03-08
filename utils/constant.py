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
element_c6 = [1.00000000e+00, 6.92199711e+01, 2.14720701e+01, # placeholder, H, He
              6.24604751e+03, 1.86461658e+03, 7.56226275e+02, 4.40658854e+02, 2.82242132e+02, 1.76224475e+02, 1.14964726e+02, 8.66654253e+01, # Li to Ne
              9.66266029e+03, 5.56711603e+03, 4.49799246e+03, 3.23170576e+03, 2.40493377e+03, 1.79471121e+03, 1.25899491e+03, 8.90770544e+02, # Na to Ar
              2.59943123e+04, 1.68207660e+04, 8.26273588e+03, 7.77095586e+03, 6.69641292e+03, 4.59291831e+03, # K to Cr
              4.39271718e+03, 3.26533023e+03, 3.02206176e+03, 3.10171893e+03, 3.40489704e+03, 3.24257643e+03, # Mn to Zn
              4.38913208e+03, 3.72928104e+03, 3.38141640e+03, 2.93024731e+03, 2.31739828e+03, 1.79682631e+03, # Ga to Kr
              3.28309165e+04, 2.45857761e+04, 1.60250923e+04, 1.15683477e+04, 1.04211383e+04, 7.35741913e+03, # Rb to Mo
              8.30215654e+03, 5.30637660e+03, 5.54485349e+03, 5.08332838e+03, 4.69068416e+03, 4.89766786e+03, # Tc to Cd
              7.19873456e+03, 6.56806354e+03, 6.30164842e+03, 5.82938112e+03, 4.89470740e+03, 3.99902045e+03, # In to Xe
              4.79920313e+04, 4.12770094e+04, 2.67154605e+04, 9.48055073e+03, 2.73610826e+04, 2.52158636e+04, 2.37894002e+04, 2.30619613e+04, # Cs to Sm
              2.20685394e+04, 1.54504314e+04, 1.88909042e+04, 1.81385061e+04, 1.77747665e+04, 1.69156794e+04, 1.63657468e+04, 1.66784756e+04, # Eu to Yb
              1.20977874e+04, 1.12542388e+04, 1.00159677e+04, 7.81963519e+03, 7.59000988e+03, 6.19523321e+03, 5.61993324e+03, 4.92110001e+03, # Lu to Pt
              4.53932590e+03, 4.53557315e+03, 8.15843304e+03, 7.89492382e+03, 7.95166128e+03, 7.73805045e+03, 6.74533997e+03, 5.68841752e+03] # Au to Rn
# element parameter r0 
element_r0 = [1.0000, 2.1823, 1.7347, # placeholder, H, He
              3.4956, 3.0982, 3.2160, 2.9103, 2.6225, 2.4817, 2.2996, 2.1374, # Li to Ne
              3.7082, 3.4839, 4.0106, 3.7917, 3.5017, 3.3107, 3.1046, 2.9148, # Na to Ar
              4.2411, 4.1035, 3.8903, 3.7642, 3.7211, 3.4414, # K to Cr 
              3.5462, 3.4421, 3.4327, 3.3462, 3.3008, 3.2309, # Mn to Zn 
              3.9579, 3.8619, 3.6625, 3.5268, 3.3662, 3.2096, # Ga to Kr
              4.6176, 4.4764, 4.2196, 4.0597, 3.8596, 3.7543, # Rb to Mo 
              3.5690, 3.4623, 3.3975, 3.3525, 3.3308, 3.4620, # Tc to Cd 
              4.2623, 4.1874, 4.0150, 3.8901, 3.7380, 3.5889, # In to Xe
              5.0567, 5.1814, 4.6261, 4.6201, 4.5702, 4.5271, 4.4896, 4.4515, # Cs to Sm 
              4.4234, 4.1243, 4.2427, 4.1541, 4.2794, 4.2450, 4.2208, 4.1986, # Eu to Yb
              4.0130, 4.2450, 4.0980, 3.9855, 3.8955, 3.7490, 3.4456, 3.3525, # Lu to Pt
              3.2564, 3.3599, 4.3127, 4.2764, 4.1175, 4.0054, 3.8644, 3.7216] # Au to Rn