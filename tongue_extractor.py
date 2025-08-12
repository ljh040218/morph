import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import ndimage
from skimage import filters, segmentation, measure
import json

# sagittal_tongue_landmarks.py에서 SagittalTongueLandmarks 클래스 import
from sagittal_tongue_landmarks import SagittalTongueLandmarks

class SagittalTongueExtractor:
    def __init__(self):
        self.debug_images = {}
        # SagittalTongueLandmarks 인스턴스 생성
        self.landmark_extractor = SagittalTongueLandmarks()
        
    def extract_tongue_sagittal(self, image_path, method='multi_approach', debug=True):
        """
        측면 뷰에서 혀 윤곽선 추출
        method: 'color', 'edge', 'clustering', 'multi_approach'
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
            
        original = image.copy()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if method == 'multi_approach':
            return self._multi_approach_extraction(image, original, debug)
        elif method == 'edge':
            return self._edge_based_extraction(image, original, debug)
        elif method == 'clustering':
            return self._clustering_based_extraction(image, original, debug)
        else:  # color
            return self._color_based_extraction(image, original, debug)
    
    def extract_with_landmarks(self, image_path, method='multi_approach', debug=True):
        """
        윤곽선 추출과 동시에 해부학적 특징점도 추출하는 통합 메서드
        """
        # 1. 먼저 윤곽선 추출
        contour, mask = self.extract_tongue_sagittal(image_path, method, debug)
        
        if contour is None:
            print("윤곽선 추출 실패로 인해 특징점 추출을 진행할 수 없습니다.")
            return None, None, None
        
        # 2. 추출된 윤곽선을 사용하여 특징점 추출
        try:
            landmarks = self.landmark_extractor.extract_sagittal_landmarks(
                image_path, contour=contour, debug=debug
            )
            
            print(f"윤곽선과 특징점 추출 완료!")
            print(f"- 윤곽선 점 개수: {len(contour)}")
            print(f"- 특징점 개수: {len(landmarks)}")
            
            return contour, mask, landmarks
            
        except Exception as e:
            print(f"특징점 추출 중 오류 발생: {e}")
            return contour, mask, None
    
    def save_complete_data(self, image_path, phoneme_name, output_dir="./output/"):
        """
        윤곽선과 특징점을 모두 추출하여 저장하는 통합 메서드
        """
        import os
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 통합 추출 실행
        contour, mask, landmarks = self.extract_with_landmarks(image_path, debug=True)
        
        if contour is None:
            print("데이터 추출 실패")
            return None
        
        # 이미지 정보
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        
        # 윤곽선 데이터 준비
        contour_coords = []
        for point in contour:
            x, y = point[0]
            # 정규화된 좌표
            contour_coords.append([float(x/w), float(y/h)])
        
        # 통합 데이터 구조
        complete_data = {
            'phoneme': phoneme_name,
            'view_type': 'sagittal',
            'image_info': {
                'path': image_path,
                'size': [w, h]
            },
            'contour': {
                'points': contour_coords,
                'point_count': len(contour_coords),
                'area': float(cv2.contourArea(contour))
            },
            'landmarks': landmarks if landmarks else {},
            'extraction_info': {
                'contour_method': 'multi_approach',
                'landmark_method': 'anatomical_sagittal'
            }
        }
        
        # JSON 파일로 저장
        output_file = os.path.join(output_dir, f"{phoneme_name}_sagittal_complete.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(complete_data, f, indent=2, ensure_ascii=False)
        
        print(f"통합 데이터 저장 완료: {output_file}")
        
        # 특징점만 별도로도 저장 (기존 메서드 활용)
        if landmarks:
            landmark_file = os.path.join(output_dir, f"{phoneme_name}_landmarks_only.json")
            self.landmark_extractor.save_sagittal_landmarks(
                landmarks, phoneme_name, (w, h), landmark_file
            )
        
        return complete_data
    
    def visualize_complete_result(self, image_path, contour, landmarks=None):
        """
        윤곽선과 특징점을 함께 시각화
        """
        image = cv2.imread(image_path)
        if image is None:
            return
        
        plt.figure(figsize=(16, 8))
        
        # 원본 이미지 + 윤곽선
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if contour is not None:
            contour_points = contour.reshape(-1, 2)
            plt.plot(contour_points[:, 0], contour_points[:, 1], 
                    'g-', linewidth=3, label='혀 윤곽선')
        
        plt.title('윤곽선 추출 결과')
        plt.axis('off')
        plt.legend()
        
        # 특징점 포함 시각화
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if contour is not None:
            plt.plot(contour_points[:, 0], contour_points[:, 1], 
                    'b-', alpha=0.6, linewidth=2, label='혀 윤곽선')
        
        if landmarks:
            # 영역별 색상 정의
            region_colors = {
                'tip': 'red', 'blade': 'orange', 'front': 'yellow',
                'middle': 'green', 'back': 'blue', 'root': 'purple',
                'peak': 'magenta', 'valley': 'cyan', 'curvature': 'brown',
                'boundary': 'pink'
            }
            
            for name, data in landmarks.items():
                coords = data['coords']
                confidence = data['confidence']
                
                # 영역에 따른 색상
                region = self.landmark_extractor.landmark_definitions.get(name, {}).get('region', 'unknown')
                color = region_colors.get(region, 'gray')
                
                # 신뢰도에 따른 크기
                size = 30 + confidence * 70
                
                plt.scatter(coords[0], coords[1], c=color, s=size, 
                          alpha=0.8, edgecolors='black', linewidth=1, zorder=5)
                
                # 라벨
                plt.annotate(name, (coords[0], coords[1]), 
                           xytext=(3, 3), textcoords='offset points',
                           fontsize=7, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.6))
        
        plt.title('윤곽선 + 해부학적 특징점')
        plt.axis('off')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def _multi_approach_extraction(self, image, original, debug=True):
        """여러 방법을 조합한 추출"""
        height, width = image.shape[:2]
        
        # 1. 색상 기반 추출
        color_mask = self._get_color_mask(image)
        
        # 2. 엣지 기반 추출  
        edge_mask = self._get_edge_mask(image)
        
        # 3. 텍스처 기반 추출
        texture_mask = self._get_texture_mask(image)
        
        # 4. 위치 기반 필터링 (구강 내부 영역에 집중)
        position_mask = self._get_position_mask(image.shape[:2])
        
        # 마스크들을 조합
        combined_mask = cv2.bitwise_and(color_mask, position_mask)
        combined_mask = cv2.bitwise_or(combined_mask, 
                                     cv2.bitwise_and(edge_mask, position_mask))
        combined_mask = cv2.bitwise_or(combined_mask,
                                     cv2.bitwise_and(texture_mask, position_mask))
        
        # 후처리
        final_mask = self._post_process_mask(combined_mask)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("윤곽선을 찾을 수 없습니다.")
            return None, None
        
        # 가장 큰 윤곽선을 혀로 선택
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 면적이 너무 작으면 제외
        min_area = (width * height) * 0.01  # 전체 이미지의 1% 이상
        if cv2.contourArea(largest_contour) < min_area:
            print(f"윤곽선 면적이 너무 작습니다: {cv2.contourArea(largest_contour)}")
            return None, None
        
        if debug:
            self._debug_visualization(original, {
                'color_mask': color_mask,
                'edge_mask': edge_mask, 
                'texture_mask': texture_mask,
                'position_mask': position_mask,
                'combined_mask': combined_mask,
                'final_mask': final_mask
            }, largest_contour)
        
        return largest_contour, final_mask
    
    def _get_color_mask(self, image):
        """색상 기반 마스크 (개선된 버전)"""
        # HSV와 LAB 색공간 모두 사용
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # 혀의 분홍/빨간색 범위 (더 넓게)
        lower_hsv1 = np.array([0, 20, 20])
        upper_hsv1 = np.array([30, 255, 255])
        lower_hsv2 = np.array([150, 20, 20])  
        upper_hsv2 = np.array([180, 255, 255])
        
        mask_hsv1 = cv2.inRange(hsv, lower_hsv1, upper_hsv1)
        mask_hsv2 = cv2.inRange(hsv, lower_hsv2, upper_hsv2)
        hsv_mask = cv2.bitwise_or(mask_hsv1, mask_hsv2)
        
        # LAB에서 a채널 활용 (빨간색 성분)
        a_channel = lab[:, :, 1]
        lab_mask = cv2.inRange(a_channel, 130, 255)  # 빨간색 성분이 높은 영역
        
        # 두 마스크 결합
        color_mask = cv2.bitwise_or(hsv_mask, lab_mask)
        
        return color_mask
    
    def _get_edge_mask(self, image):
        """엣지 기반 마스크"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 가우시안 블러로 노이즈 제거
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 여러 엣지 검출 방법 조합
        canny = cv2.Canny(blurred, 30, 100)
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_combined = np.uint8(sobel_combined / sobel_combined.max() * 255)
        
        # 엣지 마스크 결합
        edge_mask = cv2.bitwise_or(canny, cv2.threshold(sobel_combined, 50, 255, cv2.THRESH_BINARY)[1])
        
        # 형태학적 연산으로 엣지 연결
        kernel = np.ones((3, 3), np.uint8)
        edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
        edge_mask = cv2.dilate(edge_mask, kernel, iterations=2)
        
        return edge_mask
    
    def _get_texture_mask(self, image):
        """텍스처 기반 마스크"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 가버 필터로 텍스처 검출
        kernel_size = 21
        sigma = 3
        theta = 0  # 수평 방향
        lambda_val = 10
        gamma = 0.5
        
        kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lambda_val, gamma, 0, ktype=cv2.CV_32F)
        gabor_response = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        
        # 임계값 적용
        _, texture_mask = cv2.threshold(gabor_response, 50, 255, cv2.THRESH_BINARY)
        
        return texture_mask
    
    def _get_position_mask(self, image_shape):
        """위치 기반 마스크 (구강 내부 영역)"""
        height, width = image_shape
        
        # 측면 뷰에서 구강 내부 영역 추정
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 대략적인 구강 내부 영역 (이미지 중앙 하단 2/3 영역)
        start_y = height // 4
        end_y = height - height // 6
        start_x = width // 6  
        end_x = width - width // 6
        
        mask[start_y:end_y, start_x:end_x] = 255
        
        # 타원형 마스크로 더 자연스럽게
        center = (width // 2, int(height * 0.6))
        axes = (width // 3, height // 3)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        return mask
    
    def _post_process_mask(self, mask):
        """마스크 후처리"""
        # 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 가우시안 블러로 부드럽게
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # 이진화 재적용
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 홀 채우기
        mask_filled = ndimage.binary_fill_holes(mask).astype(np.uint8) * 255
        
        return mask_filled
    
    def _debug_visualization(self, original, masks, contour):
        """디버그용 시각화"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.flatten()
        
        # 원본 이미지
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title('원본 이미지')
        axes[0].axis('off')
        
        # 각종 마스크들
        mask_titles = ['색상 마스크', '엣지 마스크', '텍스처 마스크', 
                      '위치 마스크', '결합 마스크', '최종 마스크']
        mask_keys = ['color_mask', 'edge_mask', 'texture_mask',
                    'position_mask', 'combined_mask', 'final_mask']
        
        for i, (key, title) in enumerate(zip(mask_keys, mask_titles)):
            axes[i+1].imshow(masks[key], cmap='gray')
            axes[i+1].set_title(title)
            axes[i+1].axis('off')
        
        # 최종 결과
        result = original.copy()
        if contour is not None:
            cv2.drawContours(result, [contour], -1, (0, 255, 0), 3)
        axes[7].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[7].set_title('최종 윤곽선')
        axes[7].axis('off')
        
        # 빈 서브플롯 숨기기
        axes[8].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def interactive_threshold_adjustment(self, image_path):
        """대화형 임계값 조정 도구"""
        image = cv2.imread(image_path)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        def nothing(val):
            pass
        
        # 윈도우 생성
        cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
        
        # 트랙바 생성
        cv2.createTrackbar('H Min', 'Controls', 0, 179, nothing)
        cv2.createTrackbar('S Min', 'Controls', 30, 255, nothing)
        cv2.createTrackbar('V Min', 'Controls', 30, 255, nothing)
        cv2.createTrackbar('H Max', 'Controls', 30, 179, nothing)
        cv2.createTrackbar('S Max', 'Controls', 255, 255, nothing)
        cv2.createTrackbar('V Max', 'Controls', 255, 255, nothing)
        
        cv2.createTrackbar('Blur', 'Controls', 5, 21, nothing)
        cv2.createTrackbar('Morph', 'Controls', 5, 15, nothing)
        
        while True:
            # 트랙바 값 읽기
            h_min = cv2.getTrackbarPos('H Min', 'Controls')
            s_min = cv2.getTrackbarPos('S Min', 'Controls')
            v_min = cv2.getTrackbarPos('V Min', 'Controls')
            h_max = cv2.getTrackbarPos('H Max', 'Controls')
            s_max = cv2.getTrackbarPos('S Max', 'Controls')
            v_max = cv2.getTrackbarPos('V Max', 'Controls')
            
            blur_val = cv2.getTrackbarPos('Blur', 'Controls')
            morph_val = cv2.getTrackbarPos('Morph', 'Controls')
            
            # 마스크 생성
            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])
            mask = cv2.inRange(hsv, lower, upper)
            
            # 후처리
            if blur_val > 0:
                blur_val = blur_val if blur_val % 2 == 1 else blur_val + 1
                mask = cv2.GaussianBlur(mask, (blur_val, blur_val), 0)
                
            if morph_val > 0:
                kernel = np.ones((morph_val, morph_val), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 결과 표시
            result = cv2.bitwise_and(image, image, mask=mask)
            
            # 윤곽선 찾기 및 표시
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                cv2.drawContours(result, [largest], -1, (0, 255, 0), 2)
            
            # 이미지 크기 조정해서 표시
            display_img = cv2.resize(image, (400, 300))
            display_mask = cv2.resize(mask, (400, 300))
            display_result = cv2.resize(result, (400, 300))
            
            cv2.imshow('Original', display_img)
            cv2.imshow('Mask', display_mask)
            cv2.imshow('Result', display_result)
            
            # ESC로 종료
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        cv2.destroyAllWindows()
        
        return {
            'hsv_lower': [h_min, s_min, v_min],
            'hsv_upper': [h_max, s_max, v_max],
            'blur': blur_val,
            'morph': morph_val
        }

# 사용 예시
if __name__ == "__main__":
    # 통합된 추출기 사용
    extractor = SagittalTongueExtractor()
    
    # 이미지 경로
    image_path = "sagittal_tongue.png"  # 실제 이미지 경로로 변경
    
    try:
        print("=== 통합 추출 시작 ===")
        
        # 방법 1: 윤곽선과 특징점을 함께 추출
        contour, mask, landmarks = extractor.extract_with_landmarks(
            image_path, method='multi_approach', debug=True
        )
        
        if contour is not None:
            # 결과 시각화
            extractor.visualize_complete_result(image_path, contour, landmarks)
            
            # 방법 2: 완전한 데이터를 저장
            complete_data = extractor.save_complete_data(
                image_path, phoneme_name="sample_phoneme", output_dir="./output/"
            )
            
            if complete_data:
                print(f"\n=== 추출 완료 ===")
                print(f"윤곽선 점 개수: {complete_data['contour']['point_count']}")
                print(f"특징점 개수: {len(complete_data['landmarks'])}")
                print(f"데이터 저장 위치: ./output/")
        
        else:
            print("추출 실패. 대화형 도구를 사용해보세요:")
            print("extractor.interactive_threshold_adjustment('your_image.png')")
            
    except Exception as e:
        print(f"오류 발생: {e}")
        print("파일 경로와 의존성을 확인해주세요.")