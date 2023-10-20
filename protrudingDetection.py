import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

maxYVec = 0
maxXVec = 0
YVecs = []
XVecs = []

def detect_optical_flow(image_sequence):
    global maxYVec
    global maxXVec
    global YVecs
    global XVecs
    print(len(image_sequence))
    # Convert the first frame to grayscale
    prev_frame = cv2.cvtColor(image_sequence[0], cv2.COLOR_BGR2GRAY)

    # Create an empty array to store the flow vectors
    flow_vectors = []

    # Loop through the image sequence starting from the second frame
    for i in range(1, len(image_sequence)):
        # Convert the current frame to grayscale
        curr_frame = cv2.cvtColor(image_sequence[i], cv2.COLOR_BGR2GRAY)

        # Calculate the optical flow using Lucas-Kanade method
        flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        for row in flow:
            for tup in row:
                if abs(tup[0]) > maxXVec:
                    maxXVec = abs(tup[0])
                if abs(tup[1]) > maxYVec:
                    maxYVec = abs(tup[1])
                if abs(tup[0]) != 0:
                    XVecs.append(abs(tup[0]))
                if abs(tup[1]) != 0:
                    YVecs.append(abs(tup[1]))
        # Append the flow vectors to the list
        flow_vectors.append(flow)

        # Set the current frame as the previous frame for the next iteration
        prev_frame = curr_frame

    return flow_vectors

def visualize_flow(image, flow, step=16, scale=5):
    h, w = flow.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y, x].T * scale
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    for (x1, y1), (x2, y2) in lines:
        cv2.arrowedLine(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.circle(image, (x2, y2), 1, (0, 255, 0), -1)

    return image

def draw_histogram(data, bins=10):
    # Create a histogram of the data
    plt.hist(data, bins=bins, color='blue', alpha=0.7)

    # Add labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Frequency Histogram')

    # Display the histogram
    plt.show()

# Example usage:
# Replace 'path_to_frames/' with the path to your image sequence frames
# Ensure the frames are in numerical order (e.g., frame0001.jpg, frame0002.jpg, ...)
# You can use 'sorted()' to sort the frames correctly if needed.

image_sequence = [cv2.imread(f'./images/frames/frame{i}.png') for i in range(500, 750, 10)]  # Replace 10 with the number of frames in your sequence
flow_vectors = detect_optical_flow(image_sequence)
print(flow_vectors[0][0][:][0])
draw_histogram(random.sample(YVecs, 100000))
draw_histogram(random.sample(XVecs, 100000))
# Visualization
for i in range(0, len(image_sequence) - 1):
    flow_visualized = visualize_flow(image_sequence[i].copy(), flow_vectors[i])
    cv2.imshow('Optical Flow', flow_visualized)
    if cv2.waitKey(200) & 0xFF == 27:  # Press 'Esc' to exit
        break
    Image.fromarray(flow_visualized).save('./images/results/flow' + str(i) + '.png')
print(maxXVec, maxYVec)
cv2.destroyAllWindows()