import cv2
from random import randrange

# Loading some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

##################################################################
# PART 1: FOR REAL TIME IMPLEMENTATION: Use the WebCam
##################################################################
webcam = cv2.VideoCapture(0)

# ITERATE FOREVER OVER FRAMES TILL TERMINATION:
while True:

    # Read the current frame from the webcam
    successful_frame_read, frame = webcam.read()

    # Convert the image to grayscale to reduce the risk of error due to difference in color of different faces
    # ( ALTER THE VARIABLE "img" PERTAINING TO THE INPUT )
    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces using the haar cascade algorithm
    face_coordinates = trained_face_data.detectMultiScale(grayscale_image)

    # Opencv prints the attributes of the rectangle to be plotted in the form: (x_top_left, y_top_left, rect_width, rect_height)
    # Hence the array output is [[x, y, w, h]]
    # print(face_coordinates)

    # Draw the rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      color=(randrange(100, 255), randrange(100, 255), randrange(100, 255)), thickness=2)

    # Output screen with detected face
    cv2.imshow("Face Detection ML App", frame)

    # Listen for a keypress for 1ms, then move on:
    key = cv2.waitKey(1)

    # Terminate the program when Q is pressed:
    if key == 81 or key == 113:
        break

# Release the webcam
webcam.release()

print("Code completed...\nApp terminated successfully!")

##################################################################
# PART 2: TO WORK ON IMAGES: Choose an image to detect faces in
##################################################################
# img = cv2.imread('RDJ.jpg')
# grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# face_coordinates = trained_face_data.detectMultiScale(grayscale_image)
# cv2.rectangle(img, (x, y), (x + w, y + h), color=(randrange(100, 255), randrange(100, 255), randrange(100, 255)), thickness=2)
# cv2.imshow("Face Detection ML App", img)
# cv2.waitKey()
