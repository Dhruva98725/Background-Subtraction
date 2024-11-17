import cv2
import numpy as np

def frame_differencing(video_path):
    # Function for trackbar callback
    def nothing(x):
        pass

    # Naming the trackbar window
    cv2.namedWindow('Difference')

    # Creating trackbars for threshold values
    cv2.createTrackbar('Min Value', 'Difference', 0, 255, nothing)
    cv2.createTrackbar('Max Value', 'Difference', 0, 255, nothing)

    # Creating video element
    cap = cv2.VideoCapture(video_path)
    # cap = cv2.VideoCapture('thunder.mp4')
    # cap = cv2.VideoCapture('thunder2.mp4')

    # Capturing the first frame
    _, frame = cap.read()

    # Converting the first frame to grayscale
    image1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Getting the shape of the frame to create a result array for differences
    rows, cols = image1.shape
    res = np.zeros([rows, cols, 1], np.uint8)

    # Converting integers 255 and 0 to uint8 type
    a = np.uint8([255])
    b = np.uint8([0])

    # Retrieve the video's original frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(100 / fps)  # Convert FPS to milliseconds for real-time playback

    while True:
        # Read the next frame
        _, image2 = cap.read()
        if not _:  # Break if no frame is captured
            break

        # Convert the current frame to grayscale
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Get threshold values from the trackbars
        min_val = cv2.getTrackbarPos('Min Value', 'Difference')
        max_val = cv2.getTrackbarPos('Max Value', 'Difference')

        # Compute the absolute difference
        res = cv2.absdiff(image1, image2)
        cv2.imshow('Original Difference', res)

        # Create a mask based on the threshold values
        res = np.where((min_val < res) & (res < max_val), a, b).astype(np.uint8)

        # Apply the mask to the current frame
        masked_res = cv2.bitwise_and(image2, image2, mask=res)
        cv2.imshow('Difference', masked_res)

        # Update the previous frame
        image1 = image2

        # Break the loop on pressing 'Esc' key
        if cv2.waitKey(delay) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def gaussian_mixture_background_subtraction(video_path):
    def norm_pdf(x, mean, sigma):
        return (1 / (np.sqrt(2 * np.pi) * sigma)) * (np.exp(-0.5 * (((x - mean) / sigma) ** 2)))

    # Initialize the video capture
    cap = cv2.VideoCapture(video_path)
    # cap = cv2.VideoCapture('thunder.mp4')  # Uncomment to use different videos
    # cap = cv2.VideoCapture('thunder2.mp4')

    # Read the first frame and convert to grayscale
    _, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get frame dimensions
    rows, cols = frame_gray.shape

    # Initialize mean, variance, and omega matrices for the three Gaussian models
    mean = np.zeros([3, rows, cols], np.float64)
    mean[1, :, :] = frame_gray

    variance = np.zeros([3, rows, cols], np.float64)
    variance[:, :, :] = 400

    omega = np.zeros([3, rows, cols], np.float64)
    omega[0, :, :], omega[1, :, :], omega[2, :, :] = 0, 0, 1

    omega_by_sigma = np.zeros([3, rows, cols], np.float64)

    # Initialize foreground and background images
    foreground = np.zeros([rows, cols], np.uint8)
    background = np.zeros([rows, cols], np.uint8)

    # Set alpha and threshold values
    alpha = 0.3
    T = 0.5

    # Conversion for 0 and 255 values
    a = np.uint8([255])
    b = np.uint8([0])

    # Main processing loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = frame_gray.astype(np.float64)

        # Prevent negative variance values
        variance[0][np.where(variance[0] < 1)] = 10
        variance[1][np.where(variance[1] < 1)] = 5
        variance[2][np.where(variance[2] < 1)] = 1

        # Calculate the standard deviation (sigma)
        sigma1 = np.sqrt(variance[0])
        sigma2 = np.sqrt(variance[1])
        sigma3 = np.sqrt(variance[2])

        # Calculate the absolute difference between the current frame and each Gaussian mean
        compare_val_1 = cv2.absdiff(frame_gray, mean[0])
        compare_val_2 = cv2.absdiff(frame_gray, mean[1])
        compare_val_3 = cv2.absdiff(frame_gray, mean[2])

        value1 = 2.5 * sigma1
        value2 = 2.5 * sigma2
        value3 = 2.5 * sigma3

        # Identify foreground pixels based on omega and threshold T
        fore_index1 = np.where(omega[2] > T)
        fore_index2 = np.where(((omega[2] + omega[1]) > T) & (omega[2] < T))

        # Identify indices where the pixel values fit at least one Gaussian distribution
        gauss_fit_index1 = np.where(compare_val_1 <= value1)
        gauss_not_fit_index1 = np.where(compare_val_1 > value1)

        gauss_fit_index2 = np.where(compare_val_2 <= value2)
        gauss_not_fit_index2 = np.where(compare_val_2 > value2)

        gauss_fit_index3 = np.where(compare_val_3 <= value3)
        gauss_not_fit_index3 = np.where(compare_val_3 > value3)

        # Combining the indices of foreground pixels and fitting pixels for Gaussian 3
        temp = np.zeros([rows, cols])
        temp[fore_index1] = 1
        temp[gauss_fit_index3] = temp[gauss_fit_index3] + 1
        index3 = np.where(temp == 2)

        # Combining indices for foreground and fitting pixels for both Gaussian 2 and 3
        temp = np.zeros([rows, cols])
        temp[fore_index2] = 1
        index2 = np.where((compare_val_3 <= value3) | (compare_val_2 <= value2))
        temp[index2] = temp[index2] + 1
        index2 = np.where(temp == 2)

        # Identifying non-matching indices
        match_index = np.zeros([rows, cols])
        match_index[gauss_fit_index1] = 1
        match_index[gauss_fit_index2] = 1
        match_index[gauss_fit_index3] = 1
        not_match_index = np.where(match_index == 0)

        # Update the mean, variance, and omega values for matching pixels of each Gaussian
        def update_gaussian(index, mean_idx, sigma_idx, rho, constant):
            mean[mean_idx][index] = (1 - rho) * mean[mean_idx][index] + rho * frame_gray[index]
            variance[mean_idx][index] = (1 - rho) * variance[mean_idx][index] + constant
            omega[mean_idx][index] = (1 - alpha) * omega[mean_idx][index] + alpha
            omega[mean_idx][gauss_not_fit_index1] = (1 - alpha) * omega[mean_idx][gauss_not_fit_index1]

        # Update for each Gaussian
        rho = alpha * norm_pdf(frame_gray[gauss_fit_index1], mean[0][gauss_fit_index1], sigma1[gauss_fit_index1])
        constant = rho * ((frame_gray[gauss_fit_index1] - mean[0][gauss_fit_index1]) ** 2)
        update_gaussian(gauss_fit_index1, 0, sigma1, rho, constant)

        rho = alpha * norm_pdf(frame_gray[gauss_fit_index2], mean[1][gauss_fit_index2], sigma2[gauss_fit_index2])
        constant = rho * ((frame_gray[gauss_fit_index2] - mean[1][gauss_fit_index2]) ** 2)
        update_gaussian(gauss_fit_index2, 1, sigma2, rho, constant)

        rho = alpha * norm_pdf(frame_gray[gauss_fit_index3], mean[2][gauss_fit_index3], sigma3[gauss_fit_index3])
        constant = rho * ((frame_gray[gauss_fit_index3] - mean[2][gauss_fit_index3]) ** 2)
        update_gaussian(gauss_fit_index3, 2, sigma3, rho, constant)

        # Update for non-matching pixels
        mean[0][not_match_index] = frame_gray[not_match_index]
        variance[0][not_match_index] = 200
        omega[0][not_match_index] = 0.1

        # Normalize omega
        omega = omega / np.sum(omega, axis=0)

        # Calculate omega by sigma for ordering the Gaussians
        omega_by_sigma[0] = omega[0] / sigma1
        omega_by_sigma[1] = omega[1] / sigma2
        omega_by_sigma[2] = omega[2] / sigma3

        # Sort the Gaussians based on omega by sigma
        index = np.argsort(omega_by_sigma, axis=0)
        mean = np.take_along_axis(mean, index, axis=0)
        variance = np.take_along_axis(variance, index, axis=0)
        omega = np.take_along_axis(omega, index, axis=0)

        # Convert frame_gray back to uint8 for display
        frame_gray = frame_gray.astype(np.uint8)

        # Update the background image for identified foreground pixels
        background[index2] = frame_gray[index2]
        background[index3] = frame_gray[index3]

        # Display the result
        cv2.imshow('Estimated Background', cv2.subtract(frame_gray, background))
        cv2.imshow('Grayscale Video', frame_gray)

        # Break the loop on 'Esc' key press
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def median_background_estimation(video_path):
    # Creating video element
    cap = cv2.VideoCapture(video_path)

    # Images from which the background will be estimated
    images = []

    # Taking 13 frames to estimate the background
    for i in range(13):
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        images.append(frame)

    # Getting the shape of the frame to create the background
    rows, cols = frame.shape
    background = np.median(images, axis=0).astype(np.uint8)

    # Initialize a result array
    res = np.zeros([rows, cols], np.uint8)

    # Defining threshold values for mask
    a = np.uint8([255])
    b = np.uint8([0])

    # Initializing index to replace frames in the images array
    i = 0

    # Creating kernels for morphological operations
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 6))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 4))

    # Processing video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Update the background estimation
        images[i % 13] = frame_gray
        background = np.median(images, axis=0).astype(np.uint8)

        # Compute absolute difference and thresholding
        res = cv2.absdiff(frame_gray, background)
        res = np.where(res > 20, a, b).astype(np.uint8)

        # Morphological operations
        res = cv2.morphologyEx(res, cv2.MORPH_ERODE, kernel_erode)
        res = cv2.morphologyEx(res, cv2.MORPH_DILATE, kernel_dilate)

        # Apply the mask to the original frame
        colored_res = cv2.bitwise_and(frame, frame, mask=res)

        # Display results
        cv2.imshow('Foreground Mask', colored_res)
        cv2.imshow('Estimated Background', background)

        # Break on 'Esc' key press
        if cv2.waitKey(1) & 0xFF == 27:
            break

        i += 1

    cap.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    print("1. Cars on a Highway")
    print("2. Thunder")
    video = int(input("Enter your choice: "))
    if video == 1:
        video_path = 'cars.mp4'
    elif video == 2:
        video_path = 'thunder.mp4'
    else:
        print("Invalid choice")
        exit() 

    print("1. Frame Differencing")
    print("2. Gaussian Mixture Background Subtraction")
    print("3. Median Background Estimation")
    technique = int(input("Enter your choice: "))
    if technique == 1:
        frame_differencing(video_path)
    elif technique == 2:
        gaussian_mixture_background_subtraction(video_path)
    elif technique == 3:
        median_background_estimation(video_path)
    else:
        print("Invalid choice")