import numpy as np
import matplotlib.pyplot as plt

timestamps_canny = [
    0.639287, 0.860112, 1.072628, 1.272577, 1.487396, 1.699845, 1.91086, 
    2.109865, 2.321371, 2.536181, 2.748652, 2.959172, 3.159363, 3.370003, 
    3.584505, 3.798533, 4.007027, 4.202314, 4.402876, 4.6126, 4.811557, 
    5.02174, 5.221409, 5.4301, 5.626879, 5.827595, 6.039901, 6.251984, 
    6.464103, 6.662213, 6.874671, 7.08498, 7.286685, 7.494122, 7.69409, 
    7.907366, 8.118215, 8.318681, 8.525677, 8.72399, 8.921281, 9.120082, 
    9.330632, 9.527091, 9.724279, 9.919881
]

timestamps_slic = [
    1.626088, 2.204553, 2.575377, 3.050775, 3.460596, 3.838462, 4.277652, 
    4.850770, 5.232120, 5.577206, 5.920249, 6.164748, 6.540332, 6.981422, 
    7.289803, 7.603709, 8.110910, 8.624465, 8.961822, 9.304684, 9.647872, 9.986555
]

timestamps_ww = [
    0.676718, 0.942242, 1.211715, 1.469934, 1.729196, 2.001123, 2.281044, 2.540956, 
    2.798676, 3.058341, 3.320114, 3.593223, 3.880187, 4.171350, 4.449149, 4.731945, 
    4.992327, 5.259750, 5.551047, 5.841478, 6.140816, 6.450541, 6.732217, 7.013762, 
    7.296295, 7.576633, 7.856805, 8.141578, 8.413722, 8.679906, 8.934384, 9.188926, 
    9.470971, 9.754581
]

timestamps_unet = [
    0.797126, 1.112452, 1.416574, 1.717194, 2.015369, 2.316910, 2.615873, 
    2.915313, 3.220076, 3.528552, 3.833446, 4.137901, 4.439921, 4.750266, 
    5.059100, 5.368818, 5.686429, 5.993243, 6.298436, 6.612871, 6.924578, 
    7.232624, 7.535236, 7.839009, 8.145986, 8.448509, 8.754139, 9.069597, 
    9.383589, 9.691020, 9.990761
]

frames_per_interval = 5

def calculate_fps(timestamps):
    fps = []
    for i in range(1, len(timestamps)):
        interval_duration = timestamps[i] - timestamps[i-1]
        fps.append(frames_per_interval / interval_duration)
    return fps

fps_canny = calculate_fps(timestamps_canny)
print(np.mean(fps_canny))
fps_slic = calculate_fps(timestamps_slic)
print(np.mean(fps_slic))
fps_ww = calculate_fps(timestamps_ww)
print(np.mean(fps_ww))
fps_unet = calculate_fps(timestamps_unet)
print(np.mean(fps_unet))

# Plot FPS over time for each method
plt.figure(figsize=(12, 8))
plt.plot(timestamps_canny[1:], fps_canny, marker='o', linestyle='-', label='Canny Edge Detection')
plt.plot(timestamps_slic[1:], fps_slic, marker='s', linestyle='-', label='FastSLIC')
plt.plot(timestamps_ww[1:], fps_ww, marker='^', linestyle='-', label='EM-Algorithm')
plt.plot(timestamps_unet[1:], fps_unet, marker='x', linestyle='-', label='U-Net')

plt.title("Frames per Second (FPS) over Time for Different Methods")
plt.xlabel("Time (seconds)")
plt.ylabel("FPS")
plt.legend()
plt.grid(True)
plt.show()
