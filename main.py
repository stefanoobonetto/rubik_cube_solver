import cv2
import numpy as np

face = 1

faces = [['0', '0', '0', '0', '0', '0', '0', '0', '0'] for _ in range(6)]

# faces[0] = U = upper = yellow
# faces[1] = L = left = blue 
# faces[2] = F = front = red
# faces[3] = R = right = green
# faces[4] = B = back = orange
# faces[5] = D = down = white  

#               |************|
#               |*U1**U2**U3*|
#               |************|
#               |*U4**U5**U6*|
#               |************|
#               |*U7**U8**U9*|
#               |************|
#  |************|************|************|************|
#  |*L1**L2**L3*|*F1**F2**F3*|*R1**R2**R3*|*B1**B2**B3*|
#  |************|************|************|************|
#  |*L4**L5**L6*|*F4**F5**F6*|*R4**R5**R6*|*B4**B5**B6*|
#  |************|************|************|************|
#  |*L7**L8**L9*|*F7**F8**F9*|*R7**R8**R9*|*B7**B8**B9*|
#  |************|************|************|************|
#               |************|
#               |*D1**D2**D3*|
#               |************|
#               |*D4**D5**D6*|
#               |************|
#               |*D7**D8**D9*|
#               |************|

def print_cube():
    i=0
    print("         |_________|\n")
    print("         | " + faces[i][0] + "  " + faces[i][1] + "  " + faces[i][2] + " |\n")
    print("         | " + faces[i][3] + "  " + faces[i][4] + "  " + faces[i][5] + " |\n")
    print("         | " + faces[i][6] + "  " + faces[i][7] + "  " + faces[i][8] + " |\n")
    print("_________________________________________\n")
    print("| " + faces[1][0] + "  " + faces[1][1] +"  " + faces[1][2] + " | " + faces[2][0] + "  " + faces[2][1] + "  " + faces[2][2] + " | " + faces[3][0] + "  " + faces[3][1] +"  " + faces[3][2] + " | " + faces[4][0] + "  " + faces[4][1] + "  " + faces[4][2] + " |\n")
    print("| " + faces[1][3] + "  " + faces[1][4] +"  " + faces[1][5] + " | " + faces[2][3] + "  " + faces[2][4] + "  " + faces[2][5] + " | " + faces[3][3] + "  " + faces[3][4] +"  " + faces[3][5] + " | " + faces[4][3] + "  " + faces[4][4] + "  " + faces[4][5] + " |\n")
    print("| " + faces[1][3] + "  " + faces[1][4] +"  " + faces[1][5] + " | " + faces[2][3] + "  " + faces[2][4] + "  " + faces[2][5] + " | " + faces[3][3] + "  " + faces[3][4] +"  " + faces[3][5] + " | " + faces[4][3] + "  " + faces[4][4] + "  " + faces[4][5] + " |\n")
    print("_________________________________________\n")
    i=5
    print("         | " + faces[i][0] + "  " + faces[i][1] + "  " + faces[i][2] + " |\n")
    print("         | " + faces[i][3] + "  " + faces[i][4] + "  " + faces[i][5] + " |\n")
    print("         | " + faces[i][6] + "  " + faces[i][7] + "  " + faces[i][8] + " |\n")
    print("         |_________|\n")



def getcolor(h, s, v):
    if s < 65:
        return "W"
    if  (h >= 165 and h <= 180):
        return "R"
    if (h >= 0 and h < 25):
        return "O"
    if h >= 25 and h < 65:
        return "Y"
    if h >= 65 and h < 95:
        return "G"
    if h >= 95 and h < 145:
        return "B"
    return "X"

def draw_centered_grid(img, rows, cols):
    width, height = img.shape[1], img.shape[0]
    cell_size = min(width // (2*cols), height // (2*rows))

    # print(cell_size) # 80

    center_x = width // 2
    center_y = height // 2

    start_x = center_x - cell_size * cols // 2
    start_y = center_y - cell_size * rows // 2

    # print(str(start_x) + ", " + str(start_y))   # [200, 120]

    for i in range(rows + 1):
        y = start_y + i * cell_size
        cv2.line(img, (start_x, y), (start_x + cell_size * cols, y), (0, 255, 0), 1)

    for i in range(cols + 1):
        x = start_x + i * cell_size
        cv2.line(img, (x, start_y), (x, start_y + cell_size * rows), (0, 255, 0), 1)

    return img

cap = cv2.VideoCapture(0)

def show_webcam():
    # frame_with_grid = draw_centered_grid(frame.copy(), 3, 3) #grid

    # 1 = upper = yellow
    # 2 = left = blue 
    # 3 = front = red
    # 4 = right = green
    # 5 = back = orange
    # 6 = down = white  

    text = ""
    if face == 1:
        text = "Scan the face with the yellow cell at the center (U)"
        text_color = (0, 255, 255)
    elif face == 2:
        text = "Scan the face with the blue cell at the center (L)"
        text_color = (255, 0, 0)
    elif face == 3:
        text = "Scan the face with the red cell at the center (F)"
        text_color = (0, 0, 255)
    elif face == 4:
        text = "Scan the face with the green cell at the center (R)"
        text_color = (0, 255, 0)
    elif face == 5:
        text = "Scan the face with the orange cell at the center (B)"
        text_color = (0, 102, 255)
    elif face == 6:
        text = "Scan the face with the white cell at the center (D)"
        text_color = (255, 255, 255)

    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 2)[0]

    cv2.putText(frame, text, (17, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=4, lineType=cv2.LINE_AA)

    cv2.putText(frame, text, (17, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, text_color, thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow("Rubik's cube solver", frame)


while face < 7:
    ret, frame = cap.read()

    if not ret:
        break

    show_webcam()
        
    key = cv2.waitKey(1)

    if key == ord('q'):
        
        image_without_grid = frame
        
        image_path = "faces/face" + str(face) +".jpg"
        cv2.imwrite(image_path, image_without_grid)
        print("Immagine salvata come '" + image_path + "'")

        image = cv2.imread(image_path)
        
        # cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
        # cv2.imshow("Original image", image_without_grid)
        # cv2.imshow("Original image", image)

        
        # ROI immagine (cubo di rubik)
        start_x, start_y, width, height = 200, 120, 240, 240
        # roi_grid = image[start_y:start_y + width, start_x:start_x + height]
        cell_width = width // 3
        cell_height = height // 3
        roi = []

        for i in range(1, 10):
            row = (i - 1) // 3
            col = (i - 1) % 3
            zone = image[start_y + row * cell_width:start_y + (row + 1) * cell_width, start_x + col * cell_height:start_x + (col + 1) * cell_height]

            center_x = (start_x + col * cell_height + start_x + (col + 1) * cell_height) // 2
            center_y = (start_y + row * cell_width + start_y + (row + 1) * cell_width) // 2

            square_size = 80 // 2

            new_start_x = center_x - square_size // 2
            new_start_y = center_y - square_size // 2
            new_end_x = center_x + square_size // 2
            new_end_y = center_y + square_size // 2

            square_roi = image[new_start_y:new_end_y, new_start_x:new_end_x]
            roi.append(square_roi)

        # for i, square_roi in enumerate(roi):
        #     cv2.namedWindow("ZONE " + str(i + 1), cv2.WINDOW_NORMAL)
        #     cv2.imshow("ZONE " + str(i + 1), square_roi)

        predominant_colors_list = []

        color_mapping = {
            'b': 'BLUE',
            'w': 'WHITE',
            'y': 'YELLOW',
            'o': 'ORANGE',
            'r': 'RED',
            'g': 'GREEN'
        }

        for i, zone in enumerate(roi):
            zone_height, zone_width, _ = zone.shape

            tot = zone_height * zone_width
            tot_r = np.sum(zone[:, :, 2])
            tot_g = np.sum(zone[:, :, 1])
            tot_b = np.sum(zone[:, :, 0])

            r_final = int(tot_r / tot)
            g_final = int(tot_g / tot)
            b_final = int(tot_b / tot)

            # print(f"Zone {i+1}'s RGB values: ({int(r_final)}, {int(g_final)}, {int(b_final)})")

            bgr_color = np.uint8([[[b_final, g_final, r_final]]])
            hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)

            h = hsv_color[0][0][0]
            s = hsv_color[0][0][1]
            v = hsv_color[0][0][2]

            print(f"Zone {i+1}'s h value: ({int(h)})")
            print(f"Zone {i+1}'s s value: ({int(s)})")

            predominant_color_code = getcolor(h, s, v)
            predominant_color = color_mapping.get(predominant_color_code)

            faces[face-1][i] = predominant_color_code

            print(f"Zone {i+1}'s predominant color: {predominant_color}")
        face = face +1

    if key == 27: # esc
        break

color_central_cell = {
    0 : "Yellow",
    1 : "Blue",
    2 : "Red",
    3 : "Green",
    4 : "Orange",
    5 : "White",
}

print_cube()

cap.release()
cv2.destroyAllWindows()
