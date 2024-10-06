from PIL import Image
import numpy as np
import time
import torch
import cv2
from edge import edge

time0 = time.time()

# get device to work on
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# uncomment this it you want to use CPU although you actually have GPU
# device = torch.device('cpu')

print(device)

# empty image to be subtracted
filename_avg = './frames/empty/detected_lines_20240820_141634.png'
with Image.open(filename_avg) as im:
    x_avg = np.array(im)

# loop over all images to be processed
filelist = np.loadtxt('./vertical_list.txt', dtype=str)

for filename in filelist:
    filename_in = './frames/vertical/' + filename

    # load the image
    with Image.open(filename_in) as im:
        x = np.array(im)

    # subtract "empty" image
    x = x - x_avg + 128

    # process
    y, yl, std = edge(x, device)

    print(filename_in, std)

    # write output, probabilities
    filename_out = './output/' + filename
    img = Image.fromarray(y)
    #img.save(filename_out,'png')

    # write output, linear classifier
    filename_out = './outputl/' + filename
    img = Image.fromarray(yl)
    #img.save(filename_out,'png')

    height,width,channels = x.shape
    tiled_layout = np.zeros((height, width * 2, channels), dtype=np.uint8)

    # Place images into the layout
    tiled_layout[0:height, 0:width] = np.array(im)
    tiled_layout[0:height, width:width*2] = cv2.cvtColor(yl, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(filename_out,tiled_layout)
    cv2.imshow("Detected Lines (in red)",tiled_layout)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

time1 = time.time()

print('Done. Time:', (time1-time0)/len(filelist), 'sec/image')
