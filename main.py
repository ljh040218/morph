import os
import cv2
import json

def manual_landmark_annotation_tongue_cross_section(image_path):
    """
    혀 단면 이미지에 이름이 부여된 랜드마크를 시계방향으로 표시하는 함수
    (Tongue_Tip -> Superior 1-10 -> Posterior 1-5 -> Inferior 1-5)
    """
    # 21개 랜드마크의 이름과 부위별 색상을 정의
    point_definitions = (
        [{"name": "Tongue_Tip", "color": (0, 255, 255)}] + # 노란색
        [{"name": f"Superior_{i}", "color": (0, 0, 255)} for i in range(1, 11)] + # 빨간색
        [{"name": f"Posterior_{i}", "color": (0, 255, 0)} for i in range(1, 6)] + # 초록색
        [{"name": f"Inferior_{i}", "color": (255, 0, 0)} for i in range(1, 6)]   # 파란색
    )
    total_points = len(point_definitions)
    landmarks = []
    current_point_index = 0

    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        print(f"오류: 이미지를 불러올 수 없습니다 -> '{image_path}'")
        return []

    clone = img.copy()
    cv2.namedWindow('Image')

    def click_event(event, x, y, flags, param):
        nonlocal current_point_index
        if event == cv2.EVENT_LBUTTONDOWN and current_point_index < total_points:
            point_info = point_definitions[current_point_index]
            point_name = point_info["name"]
            point_color = point_info["color"]

            # 랜드마크 정보를 딕셔너리 형태로
            landmarks.append({"name": point_name, "x": x, "y": y})
            
            # 이미지에 점, 이름 표시
            cv2.circle(img, (x, y), 3, point_color, -1)
            cv2.putText(img, str(current_point_index + 1), (x + 5, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, point_color, 1)
            cv2.imshow('Image', img)
            
            print(f"'{point_name}' 랜드마크 추가됨: ({x}, {y})")
            current_point_index += 1

            # 다음 랜드마크
            if current_point_index < total_points:
                next_point_name = point_definitions[current_point_index]["name"]
                print(f"--> 다음: '{next_point_name}' 점을 클릭하세요.")
            else:
                print("\n모든 랜드마크를 찍었습니다. 'q' 키를 눌러 종료하세요.")

    cv2.setMouseCallback('Image', click_event)

    print("=== 혀 단면 랜드마크 표시 시작 ===")
    print(f"이미지: '{image_path}'")
    print(f"총 {total_points}개의 랜드마크를 순서대로 클릭하세요.")
    print("\n'q' 키를 누르면 저장 후 종료, 'r' 키를 누르면 재시작합니다.")
    print(f"--> 시작: '{point_definitions[0]['name']}' 점을 클릭하세요.")

    while True:
        cv2.imshow('Image', img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            img = clone.copy()
            landmarks.clear()
            current_point_index = 0
            cv2.imshow('Image', img)
            print("\n=== 재시작 ===")
            print(f"--> 시작: '{point_definitions[0]['name']}' 점을 클릭하세요.")

    cv2.destroyAllWindows()

    print("\n저장된 랜드마크 정보:")
    # 보기 좋게 출력하기 위해 json 모듈 사용
    print(json.dumps(landmarks, indent=2, ensure_ascii=False))
    return landmarks

# --- 이 스크립트를 직접 실행할 때만 아래 코드가 동작합니다 ---
if __name__ == "__main__":
    
    # --- 사용자는 이 부분을 자신의 환경에 맞게 수정해야 합니다 ---
    # 랜드마크를 찍고 싶은 혀 단면 이미지 경로
    manual_annotation_image = r"C:\Users\NOW\Desktop\tongue\bcdfg\PM.png"  # 🚨 실제 이미지 경로로 변경하세요

    # 결과물을 저장할 폴더
    output_folder = r"C:\Users\NOW\Desktop\tongue\results"
    
    # --- 수동 랜드마크 표시 실행 ---
    manual_landmarks = manual_landmark_annotation_tongue_cross_section(manual_annotation_image)
    
    # 결과를 파일로 저장
    if manual_landmarks:
        os.makedirs(output_folder, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(manual_annotation_image))[0]
        landmarks_file = os.path.join(output_folder, f"{base_filename}_manual_landmarks.json")

        with open(landmarks_file, 'w', encoding='utf-8') as f:
            json.dump(manual_landmarks, f, indent=2, ensure_ascii=False)
        print(f"\n수동 랜드마크가 파일로 저장되었습니다: {landmarks_file}")
    
    print("\n모든 작업이 완료되었습니다.")