import cv2
import numpy as np
import timeit

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def gaussian_blur(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def canny(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)

def region_of_interest(image):
    height, width = image.shape[:2]
    mask = np.zeros_like(image)
    polygon = np.array([[
        (0, height),
        (width, height),
        (width, height // 2),
        (0, height // 2),
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(image, mask)

def draw_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def hough_lines(image):
    return cv2.HoughLinesP(image, rho=2, theta=np.pi/180, threshold=100, minLineLength=40, maxLineGap=5)

def lane_detection_pipeline(image):
    gray = grayscale(image)
    blur = gaussian_blur(gray)
    edges = canny(blur)
    roi = region_of_interest(edges)
    lines = hough_lines(roi)
    line_image = draw_lines(image, lines)
    return cv2.addWeighted(image, 0.8, line_image, 1, 0)

def process_image(image_path):
    image = cv2.imread(image_path)
    processed_image = lane_detection_pipeline(image)
    cv2.imshow('Lane Detection', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(input_video, output_video):
    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, 20.0, (640, 480))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = lane_detection_pipeline(frame)
        out.write(processed_frame)
        cv2.imshow('Lane Detection', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def measure_performance():
    setup_code = '''
import cv2
import numpy as np
from __main__ import lane_detection_pipeline
image = cv2.imread('test_image.jpg')
'''
    test_code = '''
lane_detection_pipeline(image)
'''
    times = timeit.repeat(setup=setup_code, stmt=test_code, repeat=3, number=10)
    print(f"Execution time: {min(times)} seconds")

if __name__ == "__main__":
    
     image_path =r'C:\Users\SIDDHANT DHAKA\Desktop\lane detection project\test_1.jpg'
     process_image(image_path)
    
    
     input_video =r'C:\Users\SIDDHANT DHAKA\Desktop\lane detection project\test_sim.mp4'
     output_video = 'output_video.avi'
     process_video(input_video, output_video)

     measure_performance()
