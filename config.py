import numpy as np
DEFAULT_CONFIG = {
    'd': 0.51,
    'visiblitity': 18,
    'space': 0.01,
    'noise': True,
    'delt': 0.01,
    'start': [-0.5,0],
    'end': [0.5,0],
    'random_start': False,
    'delta_r': 0.05,
    'function':1,
    'delr' : 0.1 ,  # this is for the discrete polar coordinate 
    'deltheta' : ((np.pi/180)*11)   # discrete angles for polar coordinate 
}
# DEFAULT_CONFIG_dis = {
#     'd': 0.51,
#     'visiblitity': 18,
#     'space': 0.01,
#     'noise': True,
#     'delt': 0.0001,
#     'start': [-0.5,0],
#     'end': [0.5,0],
#     'random_start': False,
#     'delta_r': 0.05,
#     'function':1,
#     'delr' : 0.1





# }
# DEFAULT_CONFIG = {
#     'd': 0.0001,
#     'visiblitity': 16,
#     'space': 0.05,
#     'noise': False,
#     'delt': 0.05,
#     'start': [-3,0],
#     'end': [3,0],
#     'random_start': True,
#     'delta_r': 0.05,
#     'function':2

# }