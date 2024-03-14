#!/usr/bin/env python3.10

import cv2
import numpy as np
import kociemba
import time
import pyautogui

face = 1
moves_counter = 0

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)       # default camera, change index to use different webcams
cap.set(3, frameWidth)
cap.set(4, frameHeight)
screen_width, screen_height = pyautogui.size()

# Set the position of the main window in the down-right corner of the screen
window_x = screen_width - frameWidth
window_y = screen_height - frameHeight
cv2.namedWindow("Rubik's cube solver", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Rubik's cube solver", 1000, 750)        # screen size


# description for every possible move from the solution set
rubik_moves = {
    "W" : "",
    "R": "Turn right layer one time clockwise",
    "R'": "Turn right layer one time counter-clockwise",
    "R2": "Turn right layer two times",
    "L": "Turn left layer one time clockwise",
    "L'": "Turn left layer one time counter-clockwise",
    "L2": "Turn left layer two times",
    "U": "Turn upper layer one time clockwise",
    "U'": "Turn upper layer one time counter-clockwise",
    "U2": "Turn upper layer two times",
    "D": "Turn down layer one time clockwise",
    "D'": "Turn down layer one time counter-clockwise",
    "D2": "Turn down layer two times",
    "F": "Turn front layer one time clockwise",
    "F'": "Turn front layer one time counter-clockwise",
    "F2": "Turn front layer two times",
    "B": "Turn back layer one time clockwise",
    "B'": "Turn back layer one time counter-clockwise",
    "B2": "Turn back layer two times",
    "CUBE SOLVED!!!": "CUBE SOLVED!!!"
}

# dictionary n_face : "color_central_cell"
color_central_cell = {
    0 : "y",
    1 : "b",
    2 : "r",
    3 : "g",
    4 : "o",
    5 : "w",
}

# initial color ranges, then will be changed in function of the central cell values
color_ranges = {
        "g": ([0,0,0], [0,255,255]),
        "b": ([0,0,0], [0,255,255]),
        "y": ([0,0,0], [0,255,255]),
        "w": ([0,0,229], [180,50,255]),
        "o": ([0,0,0], [0,255,255]),
        "r": ([0,0,0], [0,255,255]),
}

# for each move, the corrispondent arrow that has to be shown in the webcam feed
arrows = {
    "W" : "",
    "R": "pngs/arrows/up_arrow.png",
    "R'": "pngs/arrows/down_arrow.png",
    "R2": "pngs/arrows/down_arrow_x2.png",
    "L": "pngs/arrows/down_arrow.png",
    "L'": "pngs/arrows/up_arrow.png",
    "L2": "pngs/arrows/down_arrow_x2.png",
    "U": "pngs/arrows/left_arrow.png",
    "U'": "pngs/arrows/right_arrow.png",
    "U2": "pngs/arrows/left_arrow_x2.png",
    "D": "pngs/arrows/right_arrow.png",
    "D'": "pngs/arrows/left_arrow.png",
    "D2": "pngs/arrows/left_arrow_x2.png",
    "F": "pngs/arrows/clockwise_arrow_front.png",
    "F'": "pngs/arrows/counterclockwise_arrow_front.png",
    "F2": "pngs/arrows/clockwise_arrow_front_x2.png",
    "B": "pngs/arrows/counterclockwise_arrow_back.png",
    "B'": "pngs/arrows/clockwise_arrow_back.png",
    "B2": "pngs/arrows/clockwise_arrow_back_x2.png",
    "CUBE SOLVED!!!": "CUBE SOLVED!!!"
}

# this function inverts the dictionary d
def invert_dict(d):
    return {v: k for k, v in d.items()}


# empty function for the trackbars
def empty(a):
    pass

# trackbars for the color ranges

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 620, 240)
cv2.createTrackbar("Threshold1", "Parameters", 150, 67, empty)         # 67
cv2.createTrackbar("Threshold2", "Parameters", 255, 255, empty)        # 255
cv2.createTrackbar("Area", "Parameters", 30000, 12000, empty)          #  


# this function takes the list faces as input and returns a string that represents the cube in kociemba's convention
def faces_to_string(faces):
    
    #     U
    # L   F   R   B
    #     D     

    # U, R, F, D, L, B 
    
    cube = ''.join(faces[i][j] for i in [0, 3, 2, 5, 1, 4] for j in range(9))   # adapt to kociemba's convention
    print(cube)
    cube = cube.translate(str.maketrans('wyrogb', 'DUFBRL'))
    print(cube)
    return cube
    
# this function takes the string cube as input and returns the list of moves to solve the rubik's cube
def solve_rubik(cube):

    solution = kociemba.solve(cube)

    return solution

# this function prints cube in a human-readable way
def print_cube(faces):
    i=0
    print("         |_________|\n")
    print("         | " + faces[i][0] + "  " + faces[i][1] + "  " + faces[i][2] + " |\n")
    print("         | " + faces[i][3] + "  " + faces[i][4] + "  " + faces[i][5] + " |\n")
    print("         | " + faces[i][6] + "  " + faces[i][7] + "  " + faces[i][8] + " |\n")
    print("_________________________________________\n")
    print("| " + faces[1][0] + "  " + faces[1][1] +"  " + faces[1][2] + " | " + faces[2][0] + "  " + faces[2][1] + "  " + faces[2][2] + " | " + faces[3][0] + "  " + faces[3][1] +"  " + faces[3][2] + " | " + faces[4][0] + "  " + faces[4][1] + "  " + faces[4][2] + " |\n")
    print("| " + faces[1][3] + "  " + faces[1][4] +"  " + faces[1][5] + " | " + faces[2][3] + "  " + faces[2][4] + "  " + faces[2][5] + " | " + faces[3][3] + "  " + faces[3][4] +"  " + faces[3][5] + " | " + faces[4][3] + "  " + faces[4][4] + "  " + faces[4][5] + " |\n")
    print("| " + faces[1][6] + "  " + faces[1][7] +"  " + faces[1][8] + " | " + faces[2][6] + "  " + faces[2][7] + "  " + faces[2][8] + " | " + faces[3][6] + "  " + faces[3][7] +"  " + faces[3][8] + " | " + faces[4][6] + "  " + faces[4][7] + "  " + faces[4][8] + " |\n")
    print("_________________________________________\n")
    i=5
    print("         | " + faces[i][0] + "  " + faces[i][1] + "  " + faces[i][2] + " |\n")
    print("         | " + faces[i][3] + "  " + faces[i][4] + "  " + faces[i][5] + " |\n")
    print("         | " + faces[i][6] + "  " + faces[i][7] + "  " + faces[i][8] + " |\n")
    print("         |_________|\n")

# given a square, this function returns the average hsv color of the square
def get_avg_hsv_color(ref_square):
    hsv = cv2.cvtColor(ref_square, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    h_final = h.mean()
    s_final = s.mean()
    v_final = v.mean()

    return h_final, s_final, v_final

# given an img, this function returns the ROI (face of the cube detected) and saves the squares in the folder "squares"
def getContours(img, original, imgContour, save=True):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        minArea = cv2.getTrackbarPos("Area", "Parameters")

        if area > minArea:
            cv2.drawContours(imgContour, contours, -1, (255, 0, 255), 7)
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            vertices = np.array(approx[:4], dtype=np.int32)
            mask = np.zeros_like(img)
            cv2.fillPoly(mask, [vertices], (255, 255, 255))
            x, y, w, h = cv2.boundingRect(approx)
            roi = original[y:y+h, x:x+w]
            if save:
                cv2.imwrite(f"faces/ROI{face}.jpeg", roi)
                print("Saved " + f"faces/ROI{face}.jpeg")
            return x, y, w, h

# this function shows the webcam feed with the initial instructions for the user to start scanning the cube
def show_webcam_init(frame):
    frame_display = frame.copy()

    face_texts = {
        1: ("Scan the face with the yellow cell at the center", (0, 255, 255)),
        2: ("Scan the face with the blue cell at the center", (255, 0, 0)),
        3: ("Scan the face with the red cell at the center", (0, 0, 255)),
        4: ("Scan the face with the green cell at the center", (0, 255, 0)),
        5: ("Scan the face with the orange cell at the center", (0, 102, 255)),
        6: ("Scan the face with the white cell at the center", (255, 255, 255)),
    }

    face_colors = {
        1: "yellow",
        2: "blue",
        3: "red",
        4: "green",
        5: "orange",
        6: "white",
    }

    text, text_color = face_texts.get(face, ("", (255, 255, 255)))

    cv2.putText(frame_display, text, (17, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, text_color, thickness=2, lineType=cv2.LINE_AA)

    # insert rubik's png miniatures in down-right corner of the screen

    png_img = cv2.imread('pngs/cube/cube_' + str(face_colors[face]) + '.png', cv2.IMREAD_UNCHANGED)

    aspect_ratio = png_img.shape[1] / png_img.shape[0]
    new_width = 150
    new_height = int(new_width / aspect_ratio)

    png_img = cv2.resize(png_img, (new_width, new_height))

    frame_height, frame_width = frame_display.shape[:2]
    png_height, png_width = png_img.shape[:2]

    mask = png_img[:,:,3]
    mask_inv = cv2.bitwise_not(mask)
    bgr_img = np.zeros((frame_height, frame_width, 3), np.uint8)
    
    roi_y = frame_height - png_height - 10  
    roi_x = frame_width - png_width - 10    
    bgr_img[roi_y:roi_y+png_height, roi_x:roi_x+png_width] = png_img[:,:,:3]

    roi = frame_display[roi_y:roi_y+png_height, roi_x:roi_x+png_width]
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(bgr_img[roi_y:roi_y+png_height, roi_x:roi_x+png_width], bgr_img[roi_y:roi_y+png_height, roi_x:roi_x+png_width], mask=mask)
    dst = cv2.add(img1_bg, img2_fg)
    frame_display[roi_y:roi_y+png_height, roi_x:roi_x+png_width] = dst

    cv2.imshow("Rubik's cube solver", frame_display)
    cv2.moveWindow("Rubik's cube solver", window_x, window_y)

# this function adds the arrow to the webcam feed
def add_arrow(x, y, w, h, frame_display, text):
    rubik_moves_inv = invert_dict(rubik_moves)
    arrow = cv2.imread(arrows[rubik_moves_inv[text]], cv2.IMREAD_UNCHANGED)
    aspect_ratio = arrow.shape[1] / arrow.shape[0]
    if rubik_moves_inv[text] == "R" or rubik_moves_inv[text] == "R'" or rubik_moves_inv[text] == "R2":
        x_arrow = x + w/3*2 + w/9
        y_arrow = y + h/9
        h_arrow = h - h/9*2
        w_arrow = int(h_arrow * aspect_ratio)
    elif rubik_moves_inv[text] == "L" or rubik_moves_inv[text] == "L'" or rubik_moves_inv[text] == "L2":
        x_arrow = x + w/9
        y_arrow = y + h/9  
        h_arrow = h - h/9*2
        w_arrow = int(h_arrow * aspect_ratio)
    elif rubik_moves_inv[text] == "U" or rubik_moves_inv[text] == "U'" or rubik_moves_inv[text] == "U2":
        x_arrow = x + w/9
        y_arrow = y + h/9
        w_arrow = w - w/9*2
        h_arrow = int(w / aspect_ratio)
    elif rubik_moves_inv[text] == "D" or rubik_moves_inv[text] == "D'" or rubik_moves_inv[text] == "D2":
        x_arrow = x + w/9
        y_arrow = y + h/3*2 + h/9
        w_arrow = w - w/9*2
        h_arrow = int(w / aspect_ratio)
    elif rubik_moves_inv[text] == "F" or rubik_moves_inv[text] == "F'" or rubik_moves_inv[text] == "F2":
        h_arrow = h*2/3
        w_arrow = int(h_arrow * aspect_ratio)
        x_arrow = x + (w - w_arrow)/2
        y_arrow = y + (h - h_arrow)/2
    elif rubik_moves_inv[text] == "B" or rubik_moves_inv[text] == "B'" or rubik_moves_inv[text] == "B2":
        h_arrow = h*2/3
        w_arrow = int(h_arrow * aspect_ratio)
        x_arrow = x + (w - w_arrow)/2
        y_arrow = y + (h - h_arrow)/2
    
    h_arrow = int(h_arrow)
    w_arrow = int(w_arrow)
    x_arrow = int(x_arrow)
    y_arrow = int(y_arrow)

    arrow = cv2.resize(arrow, (w_arrow, h_arrow))

    mask = arrow[:,:,3]
    mask_inv = cv2.bitwise_not(mask)
    bgr_img = np.zeros((h, w, 3), np.uint8)
    arrow_bgr = cv2.cvtColor(arrow, cv2.COLOR_RGBA2BGR)
    roi = frame_display[int(y_arrow):int(y_arrow+h_arrow), int(x_arrow):int(x_arrow+w_arrow)]
    roi_with_arrow = cv2.bitwise_and(roi, roi, mask=mask_inv)
    arrow_with_roi = cv2.bitwise_and(arrow_bgr, arrow_bgr, mask=mask)
    combined = cv2.add(roi_with_arrow, arrow_with_roi)

    frame_display[int(y_arrow):int(y_arrow+h_arrow), int(x_arrow):int(x_arrow+w_arrow)] = combined

    return frame_display

# this function shows the webcam feed once the scan is done (solution mode)
def show_webcam_sol(frame, text, move=True):
    frame_display = frame.copy()

    text_color = (255, 255, 255)  # white 
    outline_color = (0, 0, 0)     # black
    font_scale = 1 if text != "CUBE SOLVED!!!" else 2

    if text == "CUBE SOLVED!!!":
        text_color = (8, 85, 16)  # green
    
    def put_text_with_outline(text, y, font_scale):
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, 2)[0]
        text_x = (frame_display.shape[1] - text_size[0]) // 2
        cv2.putText(frame_display, text, (text_x, y+1), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, outline_color, thickness=3, lineType=cv2.LINE_AA)
        cv2.putText(frame_display, text, (text_x - 1, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, text_color, thickness=2, lineType=cv2.LINE_AA)

    put_text_with_outline(text, 441 if text != "CUBE SOLVED!!!" else 221, font_scale)

    if text != "CUBE SOLVED!!!":
        if move:
            if text != "press q when you're ready...":
                img = frame.copy()
                imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
                imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

                threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
                threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")

                imgCanny = cv2.Canny(imgGray, threshold1, threshold2, 7)
                imgDil = cv2.dilate(imgCanny, np.ones((5, 5)), iterations=1)
                x, y, w, h = getContours(imgDil, img, img.copy(), False)

                frame_display = add_arrow(x, y, w, h, frame_display, text)
        put_text_with_outline("press q when you're done...", 471, font_scale)

    cv2.imshow("Rubik's cube solver", frame_display)

# this function processes the frame scanned by the webcam and saves the squares in the folder "squares"
def process_frame(frame, face):
    img = frame.copy()
    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")

    imgCanny = cv2.Canny(imgGray, threshold1, threshold2, 7)
    imgDil = cv2.dilate(imgCanny, np.ones((5, 5)), iterations=1)

    getContours(imgDil, img, img.copy(), True)

    roi = cv2.imread(f"faces/ROI{face}.jpeg")
    h, w, _ = roi.shape
    w_square, h_square = w // 3, h // 3
    w_square_new, h_square_new = w_square // 6, h_square // 6

    squares = [
        roi[int((row + 0.5) * h_square - h_square_new / 2):int((row + 0.5) * h_square + h_square_new / 2),
            int((column + 0.5) * w_square - w_square_new / 2):int((column + 0.5) * w_square + w_square_new / 2)]
        for row in range(3) for column in range(3)
    ]

    for i, square in enumerate(squares):
        filename = f"squares/{face}/square_{i + 1}.png"
        cv2.imwrite(filename, square)

    return face + 1


# scan the 6 faces of the rubik's cube
while face < 7:
    ret, frame = cap.read()

    if not ret:
        break

    show_webcam_init(frame)
    
    key = cv2.waitKey(1)

    if key == ord('q'):
        image_path = f"scan/face{face}.jpg"         # save the frame
        cv2.imwrite(image_path, frame)
        face = process_frame(frame, face)
    elif key == 27:  # esc
        break
square_5
# take the central squares of each face (known color values) as a reference and adjust the ranges to the lighting situation I find myself in.
reference_square = []
reference_square_h = []

for i in range(6):
    ref_square_path = f"squares/{i+1}/square_5.png"
    ref_square = cv2.imread(ref_square_path)
    reference_square.append(ref_square)
    h, s, v = get_avg_hsv_color(ref_square)
    reference_square_h.append(int(h))
    if color_central_cell[i] != "w":
        color_ranges[color_central_cell[i]] = ([h-10, 0, 0], [h+10, 255, 255])

# here I take every face's square (9 for each face) and I calculate the average hsv color of each square, then I assign a color to each square
faces = []
for i in range(6):
    face_colors = []
    for j in range(9):
        path = f"squares/{i+1}/square_{j+1}.png"
        img = cv2.imread(path)
        h, s, v = map(int, get_avg_hsv_color(img))
        h, s, v = min(h, 255), min(s, 255), min(v, 255)
        if 0 <= s <= 30:
            face_colors.append("w")
        else:
            for color, (lower, upper) in color_ranges.items():
                if lower[0] <= h <= upper[0] and lower[1] <= s <= upper[1] and lower[2] <= v <= upper[2]:
                    face_colors.append(color)
                    break
            else:
                # can't find a color, so we take the closest one
                index = min(range(len(reference_square_h)-1), key=lambda i: abs(h - reference_square_h[i]))                                
                face_colors.append(color_central_cell[index])
        # double check on the similar colors (red and orange)
        if face_colors[j] == "o" or face_colors[j] == "r":
            if ((h - reference_square_h[2]) ** 2) < ((h - reference_square_h[4]) ** 2):
                face_colors[j] = "r"
            else:   
                face_colors[j] = "o"
    faces.append(face_colors)

print("\n")
print_cube(faces)
cube = faces_to_string(faces)

solution = solve_rubik(cube).split()

solution.append("CUBE SOLVED!!!")
print(solution)

current_text = "put the red face in front of the camera"
arrow_added_time = None

# show the webcam feed with hints (arrows) for the user to solve the rubik's cube
while moves_counter < len(solution)+1:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    key = cv2.waitKey(1)

    if key == ord('q'):
        print(solution[moves_counter])
        current_text = rubik_moves[solution[moves_counter]]
        moves_counter = moves_counter + 1
        arrow_added_time = time.time()

    if arrow_added_time is None:
        show_webcam_sol(frame, current_text, False)
    elif time.time() - arrow_added_time < 3:            # show the arrow for 3 seconds
        show_webcam_sol(frame, current_text, True)
    else:
        show_webcam_sol(frame, current_text, False)

    if key == 27: # esc
        break

cap.release()
cv2.destroyAllWindows()
