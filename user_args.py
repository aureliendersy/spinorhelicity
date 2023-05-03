"""
File that is used to store all of the user defined arguments
"""

# ############### The input equation -- syntax is  ab(arg1, arg2) or sb(arg1, arg2) . Powers use ** and not ^

# Simple example
input_eq = '(-ab(1, 2)**3*ab(1, 3)*ab(3, 4)*ab(3, 5)*sb(1, 5)*sb(2, 3)**2 - ab(1, 2)**3*ab(1, 3)*ab(3, 4)*ab(4, 5)*sb(1, 4)*sb(2, 3)*sb(2, 5) - ab(1, 2)**3*ab(1, 4)*ab(3, 4)*ab(4, 5)*sb(1, 2)*sb(2, 4)*sb(4, 5) - ab(1, 2)**3*ab(1, 5)*ab(2, 3)*ab(4, 5)*sb(1, 2)*sb(2, 5)**2)/(ab(1, 3)*ab(1, 4)*ab(1, 5)**2*ab(2, 3)*ab(3, 4)*ab(4, 5)*sb(1, 2)**2*sb(1, 5))'

# 17 terms for 5 pt
#input_eq = '(ab(1,2)**2*(ab(1,2)*ab(1,5)*ab(3,4)*sb(1,3)**2*sb(1,5)**2*sb(2,4) - ab(1,2)*ab(1,5)*ab(3,4)*sb(1,3)**2*sb(1,4)*sb(1,5)*sb(2,5) - ab(1,5)*ab(2,3)*ab(3,4)*sb(1,3)*sb(1,5)**2*sb(2,3)*sb(3,4) + ab(1,5)*ab(2,3)*ab(3,4)*sb(1,3)**2*sb(1,5)*sb(2,5)*sb(3,4) + ab(1,5)*ab(2,3)*ab(3,4)*sb(1,3)*sb(1,4)*sb(1,5)*sb(2,3)*sb(3,5) - ab(1,5)*ab(2,3)*ab(3,4)*sb(1,3)**2*sb(1,5)*sb(2,4)*sb(3,5) - ab(2,3)**2*ab(4,5)*sb(1,3)*sb(1,5)*sb(2,3)**2*sb(4,5) - ab(2,3)*ab(2,4)*ab(4,5)*sb(1,4)*sb(1,5)*sb(2,3)**2*sb(4,5) - ab(2,3)*ab(2,4)*ab(4,5)*sb(1,3)*sb(1,5)*sb(2,3)*sb(2,4)*sb(4,5) - ab(2,4)**2*ab(4,5)*sb(1,4)*sb(1,5)*sb(2,3)*sb(2,4)*sb(4,5) + ab(2,3)**2*ab(4,5)*sb(1,3)**2*sb(2,3)*sb(2,5)*sb(4,5) + 2*ab(2,3)*ab(2,4)*ab(4,5)*sb(1,3)*sb(1,4)*sb(2,3)*sb(2,5)*sb(4,5) + ab(2,4)**2*ab(4,5)*sb(1,4)**2*sb(2,3)*sb(2,5)*sb(4,5) - ab(2,3)*ab(3,4)*ab(4,5)*sb(1,3)*sb(1,5)*sb(2,3)*sb(3,4)*sb(4,5) - ab(2,4)*ab(3,4)*ab(4,5)*sb(1,3)*sb(1,5)*sb(2,4)*sb(3,4)*sb(4,5) + ab(2,3)*ab(3,4)*ab(4,5)*sb(1,3)**2*sb(2,5)*sb(3,4)*sb(4,5) + ab(2,4)*ab(3,4)*ab(4,5)*sb(1,3)*sb(1,4)*sb(2,5)*sb(3,4)*sb(4,5)))/(ab(1,5)*ab(2,3)*ab(2,4)*ab(2,5)*ab(3,4)*ab(4,5)*sb(1,2)**2*sb(1,5)*sb(2,3)*sb(4,5))'


# ############## Input arguments for the evaluation
beam_size = 30
nucleus_sampling = False
nucleus_p = 0.95
temperature = 1

# Input arguments for the contrastive grouping
init_cutoff = 0.90
power_decay = 0.0

# #############  Library paths
spinors_lib_path = 'environment/Spinors-1.0'
mathematica_path = None

# #############  Models: Can specify the path directly or else download it
model_path_simplifier = None
download_path_simplifier = 'https://drive.google.com/uc?export=download&id=1EpMKQUjTguISkNJXng7KLVzaDKBj627V'

model_path_contrastive = None
download_path_contrastive = 'https://drive.google.com/uc?export=download&id=1bdPMiAFsvD33XftHMQiojxzJlG72LnN9'
