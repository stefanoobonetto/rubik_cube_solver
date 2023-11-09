import cv2
import numpy as np

def getcolor(h, s, v):
    if s < 40:
        return "w"
    if (h >= 0 and h < 2) or (h >= 170 and h <= 179):
        return "r"
    if h >= 2 and h < 25:
        return "o"
    if h >= 25 and h < 65:
        return "y"
    if h >= 65 and h < 95:
        return "g"
    if h >= 95 and h < 145:
        return "b"
    return "unknown"

# def getcolor(r, g, b):
#     if (r >= 0 and r <= 30) and (g >= 0 and g <= 30) and (b >= 120 and b <= 255):
#         return 'b'
#     elif (r >= 200 and r <= 255) and (g >= 200 and g <= 255) and (b >= 200 and b <= 255):
#         return 'w'
#     elif (r >= 200 and r <= 255) and (g >= 200 and g <= 255) and (b >= 0 and b <= 30):
#         return 'y'
#     elif (r >= 200 and r <= 255) and (g >= 80 and g <= 130) and (b >= 0 and b <= 30):
#         return 'o'
#     elif (r >= 200 and r <= 255) and (g >= 0 and g <= 30) and (b >= 0 and b <= 30):
#         return 'r'
#     elif (r >= 0 and r <= 30) and (g >= 120 and g <= 255) and (b >= 0 and b <= 30):
#         return 'g'
#     else:
#         return 'unknown'

# Funzione per disegnare una griglia 3x3 centrata
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

# Inizializza la webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_with_grid = draw_centered_grid(frame.copy(), 3, 3) #grid
    
    cv2.putText(frame_with_grid, "Show face which contain  cubies at center", (int(17), 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0))

    cv2.imshow("Webcam con Griglia 3x3", frame_with_grid)

    key = cv2.waitKey(1)
    if key == ord('q'):
        image_without_grid = frame
        cv2.imwrite("rubik_cube4.jpg", image_without_grid)
        print("Immagine senza la griglia salvata come 'rubik_cube.jpg'")

        image = cv2.imread('rubik_cube.jpg')
        
        cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
        cv2.imshow("Original image", image_without_grid)
        cv2.imshow("Original image", image)

        
        # ROI immagine (cubo di rubik)
        start_x, start_y, width, height = 200, 120, 240, 240

        roi_grid = image[start_y:start_y + 240, start_x:start_x + 240]

        roi = []

        for i in range(1, 10):
            row = (i - 1) // 3
            col = (i - 1) % 3
            zone = image[start_y + row * 80:start_y + (row + 1) * 80, start_x + col * 80:start_x + (col + 1) * 80]

            # Calcola il centro della zona esistente
            center_x = (start_x + col * 80 + start_x + (col + 1) * 80) // 2
            center_y = (start_y + row * 80 + start_y + (row + 1) * 80) // 2

            # Calcola la nuova dimensione del lato del quadrato (dimezzata)
            square_size = 80 // 2

            # Calcola i nuovi punti di inizio e fine per la ROI quadrate
            new_start_x = center_x - square_size // 2
            new_start_y = center_y - square_size // 2
            new_end_x = center_x + square_size // 2
            new_end_y = center_y + square_size // 2

            square_roi = image[new_start_y:new_end_y, new_start_x:new_end_x]
            roi.append(square_roi)

        for i, square_roi in enumerate(roi):
            cv2.namedWindow("ZONE " + str(i + 1), cv2.WINDOW_NORMAL)
            cv2.imshow("ZONE " + str(i + 1), square_roi)

        # qui sopra ho sistemato il codice in modo tale da creare una lista di ROI corrispondente ai 9 quadratini 
        # di ciascuna faccia del cubo, la fase successiva delle maschere deve essere adattata in modo tale da 
        # iterare sulla lista roi[]

        #sarebbe da provare a ritagliare la zona contornata dal nero 

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

            predominant_color_code = getcolor(h, s, v)
            predominant_color = color_mapping.get(predominant_color_code)

            print(f"Zone {i+1}'s predominant color: {predominant_color}")

# Rest of the code for displaying the results using OpenCV
    if key == 27: # esc
        break

# Rilascia la webcam e chiudi la finestra
cap.release()
cv2.destroyAllWindows()

































        # # creo maschere
        # red_mask = cv2.inRange(roi, lower_red, upper_red)
        # yellow_mask = cv2.inRange(roi, lower_yellow, upper_yellow)
        # green_mask = cv2.inRange(roi, lower_green, upper_green)
        # blue_mask = cv2.inRange(roi, lower_blue, upper_blue)
        # white_mask = cv2.inRange(roi, lower_white, upper_white)
        # orange_mask = cv2.inRange(roi, lower_orange, upper_orange)
        
        # # check 
        # if np.any(red_mask):
        #     red_detected = True

        # if np.any(yellow_mask):
        #     yellow_detected = True

        # if np.any(green_mask):
        #     green_detected = True

        # if np.any(blue_mask):
        #     blue_detected = True

        # if np.any(white_mask):
        #     white_detected = True

        # if np.any(orange_mask):
        #     orange_detected = True

        # if red_detected:
        #     print("ROSSO: presente")
        # else:
        #     print("ROSSO: non presente")

        # if yellow_detected:
        #     print("GIALLO: presente")
        # else:
        #     print("GIALLO: non presente")

        # if green_detected:
        #     print("VERDE: presente")
        # else:
        #     print("VERDE: non presente")

        # if blue_detected:
        #     print("BLU: presente")
        # else:
        #     print("BLU: non presente")

        # if white_detected:
        #     print("BIANCO: presente")
        # else:
        #     print("BIANCO: non presente")

        # if orange_detected:
        #     print("ARANCIONE: presente")
        # else:
        #     print("ARANCIONE: non presente")


        # color_mask = red_mask | yellow_mask | green_mask | blue_mask | white_mask | orange_mask
