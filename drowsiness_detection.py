import cv2
import dlib
import numpy as np
from scipy.spatial import distance

# Tính khoảng cách Euclidean
def euclidean_dist(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

# Hàm tính toán EAR (Eye Aspect Ratio) dựa trên công thức
def calculate_ear(eye):
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])
    C = euclidean_dist(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Khởi tạo detector và predictor của dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Ngưỡng EAR để xác định mắt nhắm
EAR_THRESHOLD = 0.3
FRAME_THRESHOLD = 20
frame_count = 0

# Khởi tạo video capture từ camera
cap = cv2.VideoCapture(0)

# Lấy kích thước khung hình từ camera
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Đặt tên file và codec để lưu video
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec XVID
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Lấy các điểm mốc của mắt (p1-p6 cho mắt trái và phải)
        left_eye = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                             (landmarks.part(37).x, landmarks.part(37).y),
                             (landmarks.part(38).x, landmarks.part(38).y),
                             (landmarks.part(39).x, landmarks.part(39).y),
                             (landmarks.part(40).x, landmarks.part(40).y),
                             (landmarks.part(41).x, landmarks.part(41).y)])

        right_eye = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                              (landmarks.part(43).x, landmarks.part(43).y),
                              (landmarks.part(44).x, landmarks.part(44).y),
                              (landmarks.part(45).x, landmarks.part(45).y),
                              (landmarks.part(46).x, landmarks.part(46).y),
                              (landmarks.part(47).x, landmarks.part(47).y)])

        # Vẽ viền chấm chấm quanh hốc mắt
        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)  # Mắt trái màu đỏ
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)  # Mắt phải màu đỏ

        # Tính EAR cho mỗi mắt
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Hiển thị chỉ số EAR lên màn hình
        cv2.putText(frame, f'EAR: {ear:.2f}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Kiểm tra trạng thái buồn ngủ
        if (ear < EAR_THRESHOLD):
            frame_count += 1
            if frame_count >= FRAME_THRESHOLD:
                cv2.putText(frame, "BUON NGU!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            frame_count = 0

    # Ghi khung hình vào file video
    out.write(frame)

    # Hiển thị khung hình
    cv2.imshow('Drowsiness Detection', frame)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
out.release()  # Đóng file video
cv2.destroyAllWindows()
