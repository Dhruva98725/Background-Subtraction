import cv2
import numpy as np

def frame_differencing(video_path):
    def nothing(x):
        pass

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    cv2.namedWindow('Difference')
    cv2.createTrackbar('Min Value', 'Difference', 0, 255, nothing)
    cv2.createTrackbar('Max Value', 'Difference', 0, 255, nothing)

    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read the first frame from the video.")
        cap.release()
        return

    image1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30  # Default to 30ms if FPS is invalid

    while True:
        ret, image2 = cap.read()
        if not ret:
            print("Error: Unable to read the current frame from the video.")
            cap.release()
            return

        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        min_val = cv2.getTrackbarPos('Min Value', 'Difference')
        max_val = cv2.getTrackbarPos('Max Value', 'Difference')

        res = cv2.absdiff(image1, image2)
        cv2.imshow('Original Difference', res)

        mask = ((res > min_val) & (res < max_val)).astype(np.uint8) * 255
        masked_res = cv2.bitwise_and(image2, image2, mask=mask)
        cv2.imshow('Difference', masked_res)

        image1 = image2
        if cv2.waitKey(delay) & 0xFF == 27:
            break


    cap.release()

    cv2.destroyAllWindows()

def mean_filtering(video_path):
    def nothing(x):
        pass

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    images = []

    cv2.namedWindow('tracker')
    cv2.createTrackbar('val','tracker',50,255,nothing)
    while True:
        ret,frame = cap.read()
        if not ret:
            print("Error: Unable to read the current frame from the video.")
            cap.release()
            return

        cv2.imshow('image',frame)
        dim = (500,500)
        frame = cv2.resize(frame,dim,interpolation = cv2.INTER_AREA) 
        #converting images into grayscale       
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        images.append(frame_gray)
        # removing the images after every 50 image
        if len(images)==50:
            images.pop(0)

        image = np.array(images)
        # gettting the tracker value
        val = cv2.getTrackbarPos('val','tracker')
        image = np.mean(image,axis=0)
        image = image.astype(np.uint8)
        cv2.imshow('background',image)
        # foreground will be background - curr frame
        foreground_image = cv2.absdiff(frame_gray,image)

        a = np.array([0],np.uint8)

        img = np.where(foreground_image>val,frame_gray,a)
        cv2.imshow('foreground',img)

        if cv2.waitKey(1) & 0xFF == 27:
            break


    cap.release()

    cv2.destroyAllWindows()		

def running_average(video_path):
    def nothing(x):
        pass

    # Video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    # Create trackbar window
    cv2.namedWindow('tracker')
    cv2.createTrackbar('val', 'tracker', 50, 255, nothing)
    cv2.createTrackbar('alpha', 'tracker', 5, 100, nothing)  # alpha trackbar

    # Initialize background model
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read the first frame from the video.")
        cap.release()
        return

    dim = (500, 500)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    background = frame_gray.astype(np.float32)

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read the current frame from the video.")
            cap.release()
            return


        # Resize the frame to 500x500 for easier processing
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        
        # Convert frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get trackbar values
        val = cv2.getTrackbarPos('val', 'tracker')
        alpha = cv2.getTrackbarPos('alpha', 'tracker') / 100.0  # Scale alpha to [0, 1]

        # Update the background model with running average
        cv2.accumulateWeighted(frame_gray, background, alpha)

        # Compute foreground mask
        foreground_image = cv2.absdiff(frame_gray, cv2.convertScaleAbs(background))

        # Apply thresholding to detect foreground
        _, img = cv2.threshold(foreground_image, val, 255, cv2.THRESH_BINARY)

        # Display the background and foreground images
        cv2.imshow('background', cv2.convertScaleAbs(background))
        cv2.imshow('foreground', img)

        # Break the loop if the user presses the ESC key
        if cv2.waitKey(delay) & 0xFF == 27:
            break


    cap.release()

    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    
    video_path = 'cars.mp4'

    print("Choose a technique:")
    print("1. Frame Differencing")
    print("2. Mean Filtering")
    print("3. Running Average")
    technique = int(input("Enter your choice: "))

    if technique == 1:
        frame_differencing(video_path)
    elif technique == 2:
        mean_filtering(video_path)
    elif technique == 3:
        running_average(video_path)
    else:
        print("Invalid technique choice.")
