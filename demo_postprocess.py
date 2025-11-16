import numpy as np
from utils import *
cut = 6000
demo = load_buffer_from_pkl("demonstrations_up_to_gen_BrandsHatchUNNORM.csv")
demo_new = demo[:cut].copy()
counter = 0
for i, d in enumerate(demo[:cut]):
    scan = np.array(d['scan'])
    state = np.array(d['state'])
    action = np.array(d['action'])
    print(state)
    if np.isclose(abs(action[0]), 7.639055) or np.isclose(abs(action[0]), 0.):
        counter += 1
        demo_new[i]['action'][0] = action[0] / abs(action[0]) if action[0] != 0 else 0
        new_steer = demo_new[i]['action'][0]
        if new_steer < 0:
            norm_loc = 0.05
            norm_scale = 0.05
        elif new_steer > 0:
            norm_loc = -0.05
            norm_scale = 0.05
        else:
            norm_loc = 0.
            norm_scale = 0.1
        randnorm = np.random.normal(norm_loc, norm_scale)
        demo_new[i]['action'][0] += randnorm
        demo_new[i]['action'][0] = np.clip(demo_new[i]['action'][0], -1.0, 1.0)
    # print(f"Demo {i}: State values: {state[0]:.2f} {state[1]:.2f} {state[2]:.2f} -- Action values: {action[0]:.2f} {action[1]:.2f}")
print(f"Number of clamped steering actions: {counter} out of {len(demo)}")

buffer_to_pkl(demo_new, "demonstrations_up_to_gen_BrandsHatch.pkl")