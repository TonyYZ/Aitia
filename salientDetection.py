import cv2

# Define the path for your input and output videos
input_video_path = './images/videos/irrigate.mp4'
output_video_path = './images/results/videos/irrigate salient.avi'
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error opening video stream or file")

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Initialize OpenCV's static saliency spectral residual detector
saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization to improve contrast
        equalized_frame = cv2.equalizeHist(gray_frame)

        # Compute the saliency map
        (success, saliency_map) = saliency.computeSaliency(equalized_frame)
        if not success:
            print("Failed to compute saliency map")
            break

        # Convert the saliency map from floating point to unsigned byte
        saliency_map = (saliency_map * 255).astype('uint8')

        # Optional: Apply a threshold to binarize the saliency map
        _, saliency_map_thresh = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Merge the saliency map with the original color frame for visualization
        saliency_map_color = cv2.merge([saliency_map_thresh] * 3)
        output_frame = cv2.addWeighted(frame, 0.5, saliency_map_color, 0.5, 0)

        # Write the frame with saliency overlay to the output video
        out.write(output_frame)

        # Display the frame
        cv2.imshow('Frame with Salient Regions', output_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"The video with salient regions was saved to {output_video_path}")