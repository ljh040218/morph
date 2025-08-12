import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import json

class SagittalTongueLandmarks:
    def __init__(self):
        # 측면 뷰에서 보이는 혀의 해부학적 특징점 정의
        self.landmark_definitions = {
            # 혀 상부 윤곽선 (혀등, dorsum)
            'tip_top': {'name': '혀끝_상단', 'description': '혀끝의 가장 위쪽 점', 'region': 'tip'},
            'tip_front': {'name': '혀끝_전방', 'description': '혀끝의 가장 앞쪽 점', 'region': 'tip'},
            'blade_top': {'name': '혀날_상단', 'description': '혀날 부위의 최고점', 'region': 'blade'},
            'front_dorsum': {'name': '전방_혀등', 'description': '혀등 전방부 최고점', 'region': 'front'},
            'mid_dorsum_high': {'name': '중앙_혀등_높은점', 'description': '혀등 중앙부 최고점', 'region': 'middle'},
            'mid_dorsum_low': {'name': '중앙_혀등_낮은점', 'description': '혀등 중앙부 상대적 낮은점', 'region': 'middle'},
            'back_dorsum': {'name': '후방_혀등', 'description': '혀등 후방부 최고점', 'region': 'back'},
            'root_top': {'name': '혀뿌리_상단', 'description': '혀뿌리 가장 높은 점', 'region': 'root'},
            
            # 혀 하부 윤곽선 (혀 아래쪽)
            'tip_bottom': {'name': '혀끝_하단', 'description': '혀끝의 가장 아래쪽 점', 'region': 'tip'},
            'blade_bottom': {'name': '혀날_하단', 'description': '혀날 부위 아래쪽', 'region': 'blade'},
            'front_ventral': {'name': '전방_복측', 'description': '혀 전방부 아래쪽', 'region': 'front'},
            'mid_ventral_low': {'name': '중앙_복측_낮은점', 'description': '혀 중앙부 아래쪽 최저점', 'region': 'middle'},
            'mid_ventral_high': {'name': '중앙_복측_높은점', 'description': '혀 중앙부 아래쪽 상대적 높은점', 'region': 'middle'},
            'back_ventral': {'name': '후방_복측', 'description': '혀 후방부 아래쪽', 'region': 'back'},
            'root_bottom': {'name': '혀뿌리_하단', 'description': '혀뿌리 아래쪽', 'region': 'root'},
            
            # 특별한 해부학적 지점들
            'apex': {'name': '혀첨', 'description': '혀의 가장 끝점', 'region': 'tip'},
            'sulcus_terminalis': {'name': '경계구', 'description': '혀의 전방부와 후방부 경계', 'region': 'boundary'},
            'dorsum_peak': {'name': '혀등_최고점', 'description': '전체 혀등에서 가장 높은 지점', 'region': 'peak'},
            'ventral_deepest': {'name': '복측_최저점', 'description': '혀 아래쪽에서 가장 깊은 지점', 'region': 'valley'},
            'curvature_max': {'name': '최대_곡률점', 'description': '혀 윤곽선에서 곡률이 가장 큰 지점', 'region': 'curvature'}
        }
        
        self.landmarks = {}
        
    def extract_sagittal_landmarks(self, image_path, contour=None, debug=True):
        """측면 뷰에서 해부학적 특징점 추출"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        
        # 윤곽선이 없으면 먼저 추출
        if contour is None:
            from sagittal_tongue_extractor import SagittalTongueExtractor
            extractor = SagittalTongueExtractor()
            contour, _ = extractor.extract_tongue_sagittal(image_path, debug=False)
            
            if contour is None:
                raise ValueError("혀 윤곽선을 찾을 수 없습니다.")
        
        # 윤곽선을 numpy 배열로 변환
        points = contour.reshape(-1, 2)
        
        # 이미지 정보
        height, width = image.shape[:2]
        
        # 특징점 추출
        landmarks = self._identify_sagittal_landmarks(points, width, height)
        
        self.landmarks = landmarks
        
        if debug:
            self._visualize_sagittal_landmarks(image, points, landmarks)
        
        return landmarks
    
    def _identify_sagittal_landmarks(self, points, width, height):
        """측면 뷰 특징점 식별"""
        landmarks = {}
        
        # 1. 기본 좌표 분석
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # 2. 혀를 상부와 하부로 분리
        # 혀의 중심선 추정 (y 좌표의 중간값)
        center_y = (y_min + y_max) / 2
        
        upper_points = points[points[:, 1] <= center_y]  # 상부 (혀등)
        lower_points = points[points[:, 1] > center_y]   # 하부 (복측)
        
        # 3. 혀를 전후 영역으로 분할
        x_range = x_max - x_min
        
        # 전방 (0-30%), 중앙 (30-70%), 후방 (70-100%)
        front_boundary = x_min + x_range * 0.3
        back_boundary = x_min + x_range * 0.7
        
        # 4. 주요 특징점들 추출
        
        # 혀끝 관련 점들
        tip_candidates = points[points[:, 0] <= front_boundary]
        if len(tip_candidates) > 0:
            # 가장 앞쪽 점 (apex)
            apex_idx = np.argmin(tip_candidates[:, 0])
            landmarks['apex'] = {
                'coords': tip_candidates[apex_idx],
                'confidence': 0.95,
                'description': self.landmark_definitions['apex']['description']
            }
            
            # 혀끝 상단과 하단
            tip_upper = tip_candidates[tip_candidates[:, 1] <= center_y]
            tip_lower = tip_candidates[tip_candidates[:, 1] > center_y]
            
            if len(tip_upper) > 0:
                tip_top_idx = np.argmin(tip_upper[:, 1])  # 가장 위쪽
                landmarks['tip_top'] = {
                    'coords': tip_upper[tip_top_idx],
                    'confidence': 0.9,
                    'description': self.landmark_definitions['tip_top']['description']
                }
            
            if len(tip_lower) > 0:
                tip_bottom_idx = np.argmax(tip_lower[:, 1])  # 가장 아래쪽
                landmarks['tip_bottom'] = {
                    'coords': tip_lower[tip_bottom_idx],
                    'confidence': 0.9,
                    'description': self.landmark_definitions['tip_bottom']['description']
                }
        
        # 혀 상부 (혀등) 특징점들
        if len(upper_points) > 0:
            # 전체 혀등에서 가장 높은 점
            dorsum_peak_idx = np.argmin(upper_points[:, 1])
            landmarks['dorsum_peak'] = {
                'coords': upper_points[dorsum_peak_idx],
                'confidence': 0.9,
                'description': self.landmark_definitions['dorsum_peak']['description']
            }
            
            # 영역별 혀등 점들
            self._extract_dorsum_points(upper_points, landmarks, front_boundary, back_boundary)
        
        # 혀 하부 (복측) 특징점들  
        if len(lower_points) > 0:
            # 가장 깊은 점
            ventral_deepest_idx = np.argmax(lower_points[:, 1])
            landmarks['ventral_deepest'] = {
                'coords': lower_points[ventral_deepest_idx],
                'confidence': 0.8,
                'description': self.landmark_definitions['ventral_deepest']['description']
            }
            
            # 영역별 복측 점들
            self._extract_ventral_points(lower_points, landmarks, front_boundary, back_boundary)
        
        # 곡률 기반 특징점
        curvature_points = self._find_curvature_landmarks(points)
        landmarks.update(curvature_points)
        
        # 혀뿌리 관련 점들
        root_points = self._extract_root_landmarks(points, back_boundary, center_y)
        landmarks.update(root_points)
        
        return landmarks
    
    def _extract_dorsum_points(self, upper_points, landmarks, front_boundary, back_boundary):
        """혀등 (상부) 특징점들 추출"""
        # 전방 혀등
        front_dorsum = upper_points[upper_points[:, 0] <= front_boundary]
        if len(front_dorsum) > 0:
            front_peak_idx = np.argmin(front_dorsum[:, 1])
            landmarks['front_dorsum'] = {
                'coords': front_dorsum[front_peak_idx],
                'confidence': 0.8,
                'description': self.landmark_definitions['front_dorsum']['description']
            }
        
        # 중앙 혀등 (높은점과 낮은점)
        mid_dorsum = upper_points[(upper_points[:, 0] > front_boundary) & 
                                 (upper_points[:, 0] <= back_boundary)]
        if len(mid_dorsum) > 2:
            # 중앙 영역에서 가장 높은 점과 상대적으로 낮은 점
            sorted_by_y = mid_dorsum[np.argsort(mid_dorsum[:, 1])]
            
            landmarks['mid_dorsum_high'] = {
                'coords': sorted_by_y[0],  # 가장 높은 점
                'confidence': 0.8,
                'description': self.landmark_definitions['mid_dorsum_high']['description']
            }
            
            if len(sorted_by_y) > len(sorted_by_y)//2:
                landmarks['mid_dorsum_low'] = {
                    'coords': sorted_by_y[len(sorted_by_y)//2],  # 중간 높이 점
                    'confidence': 0.6,
                    'description': self.landmark_definitions['mid_dorsum_low']['description']
                }
        
        # 후방 혀등
        back_dorsum = upper_points[upper_points[:, 0] > back_boundary]
        if len(back_dorsum) > 0:
            back_peak_idx = np.argmin(back_dorsum[:, 1])
            landmarks['back_dorsum'] = {
                'coords': back_dorsum[back_peak_idx],
                'confidence': 0.7,
                'description': self.landmark_definitions['back_dorsum']['description']
            }
    
    def _extract_ventral_points(self, lower_points, landmarks, front_boundary, back_boundary):
        """복측 (하부) 특징점들 추출"""
        # 전방 복측
        front_ventral = lower_points[lower_points[:, 0] <= front_boundary]
        if len(front_ventral) > 0:
            front_low_idx = np.argmax(front_ventral[:, 1])
            landmarks['front_ventral'] = {
                'coords': front_ventral[front_low_idx],
                'confidence': 0.7,
                'description': self.landmark_definitions['front_ventral']['description']
            }
        
        # 중앙 복측
        mid_ventral = lower_points[(lower_points[:, 0] > front_boundary) & 
                                  (lower_points[:, 0] <= back_boundary)]
        if len(mid_ventral) > 2:
            sorted_by_y = mid_ventral[np.argsort(-mid_ventral[:, 1])]  # 내림차순
            
            landmarks['mid_ventral_low'] = {
                'coords': sorted_by_y[0],  # 가장 낮은 점
                'confidence': 0.7,
                'description': self.landmark_definitions['mid_ventral_low']['description']
            }
            
            if len(sorted_by_y) > len(sorted_by_y)//2:
                landmarks['mid_ventral_high'] = {
                    'coords': sorted_by_y[len(sorted_by_y)//2],
                    'confidence': 0.6,
                    'description': self.landmark_definitions['mid_ventral_high']['description']
                }
        
        # 후방 복측
        back_ventral = lower_points[lower_points[:, 0] > back_boundary]
        if len(back_ventral) > 0:
            back_low_idx = np.argmax(back_ventral[:, 1])
            landmarks['back_ventral'] = {
                'coords': back_ventral[back_low_idx],
                'confidence': 0.7,
                'description': self.landmark_definitions['back_ventral']['description']
            }
    
    def _find_curvature_landmarks(self, points):
        """곡률 기반 특징점 찾기"""
        landmarks = {}
        
        if len(points) < 5:
            return landmarks
        
        # 곡률 계산 (3점을 이용한 근사)
        curvatures = []
        for i in range(2, len(points)-2):
            p1, p2, p3 = points[i-2], points[i], points[i+2]
            
            # 벡터 계산
            v1 = p2 - p1
            v2 = p3 - p2
            
            # 외적으로 곡률 근사
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cross_prod = np.cross(v1, v2)
                curvature = abs(cross_prod) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                curvatures.append((i, curvature, points[i]))
        
        if curvatures:
            # 가장 높은 곡률을 가진 점
            max_curvature = max(curvatures, key=lambda x: x[1])
            landmarks['curvature_max'] = {
                'coords': max_curvature[2],
                'confidence': 0.6,
                'description': self.landmark_definitions['curvature_max']['description']
            }
        
        return landmarks
    
    def _extract_root_landmarks(self, points, back_boundary, center_y):
        """혀뿌리 관련 특징점들"""
        landmarks = {}
        
        root_region_points = points[points[:, 0] > back_boundary]
        
        if len(root_region_points) > 0:
            # 혀뿌리 상단
            root_upper = root_region_points[root_region_points[:, 1] <= center_y]
            if len(root_upper) > 0:
                root_top_idx = np.argmin(root_upper[:, 1])
                landmarks['root_top'] = {
                    'coords': root_upper[root_top_idx],
                    'confidence': 0.7,
                    'description': self.landmark_definitions['root_top']['description']
                }
            
            # 혀뿌리 하단
            root_lower = root_region_points[root_region_points[:, 1] > center_y]
            if len(root_lower) > 0:
                root_bottom_idx = np.argmax(root_lower[:, 1])
                landmarks['root_bottom'] = {
                    'coords': root_lower[root_bottom_idx],
                    'confidence': 0.7,
                    'description': self.landmark_definitions['root_bottom']['description']
                }
        
        return landmarks
    
    def _visualize_sagittal_landmarks(self, image, contour_points, landmarks):
        """특징점 시각화"""
        plt.figure(figsize=(16, 10))
        
        # 원본 이미지
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 윤곽선 그리기
        plt.plot(contour_points[:, 0], contour_points[:, 1], 'b-', alpha=0.5, linewidth=2, label='혀 윤곽선')
        
        # 특징점들을 영역별로 색상 구분
        region_colors = {
            'tip': 'red',
            'blade': 'orange', 
            'front': 'yellow',
            'middle': 'green',
            'back': 'blue',
            'root': 'purple',
            'peak': 'magenta',
            'valley': 'cyan',
            'curvature': 'brown',
            'boundary': 'pink'
        }
        
        for name, data in landmarks.items():
            coords = data['coords']
            confidence = data['confidence']
            
            # 영역에 따른 색상 결정
            region = self.landmark_definitions.get(name, {}).get('region', 'unknown')
            color = region_colors.get(region, 'gray')
            
            # 신뢰도에 따른 크기 조절
            size = 30 + confidence * 100
            
            plt.scatter(coords[0], coords[1], c=color, s=size, alpha=0.8, 
                       edgecolors='black', linewidth=1, zorder=5)
            
            # 라벨 표시
            plt.annotate(name, (coords[0], coords[1]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
        
        plt.title('측면 뷰 혀 해부학적 특징점', fontsize=16, fontweight='bold')
        plt.axis('off')
        
        # 범례 생성
        legend_elements = [plt.scatter([], [], c=color, s=100, label=region, alpha=0.8) 
                          for region, color in region_colors.items()]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        plt.show()
    
    def save_sagittal_landmarks(self, landmarks, phoneme_name, image_size, output_path):
        """특징점을 JSON으로 저장"""
        width, height = image_size
        
        # 좌표 정규화
        normalized_landmarks = {}
        for name, data in landmarks.items():
            coords = data['coords']
            normalized_coords = [float(coords[0]/width), float(coords[1]/height)]
            
            normalized_landmarks[name] = {
                'coords': normalized_coords,
                'confidence': data['confidence'],
                'description': data['description'],
                'region': self.landmark_definitions.get(name, {}).get('region', 'unknown')
            }
        
        output_data = {
            'phoneme': phoneme_name,
            'view_type': 'sagittal',
            'landmarks': normalized_landmarks,
            'image_size': [width, height],
            'landmark_count': len(normalized_landmarks),
            'landmark_definitions': self.landmark_definitions
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"측면 뷰 특징점 저장 완료: {output_path}")
        return output_data

# 사용 예시
if __name__ == "__main__":
    extractor = SagittalTongueLandmarks()
    
    # 측면 뷰 이미지에서 특징점 추출
    image_path = r"C:\Users\yunha\OneDrive\바탕 화면\윤예진 파일\3학년 1학기\IT 경진대회) 바름\9주차\aeiou\ㅏ.png"
    
    try:
        landmarks = extractor.extract_sagittal_landmarks(image_path, debug=True)
        
        if landmarks:
            print(f"\n추출된 특징점: {len(landmarks)}개")
            
            # 영역별 특징점 개수
            regions = {}
            for name, data in landmarks.items():
                region = extractor.landmark_definitions.get(name, {}).get('region', 'unknown')
                regions[region] = regions.get(region, 0) + 1
            
            print("\n영역별 특징점:")
            for region, count in regions.items():
                print(f"  {region}: {count}개")
            
            # 이미지 크기 (실제 이미지에서 가져오기)
            image = cv2.imread(image_path)
            h, w = image.shape[:2]
            
            # 저장
            extractor.save_sagittal_landmarks(landmarks, "sample_phoneme", (w, h), "sagittal_landmarks.json")
            
        else:
            print("특징점 추출 실패")
            
    except Exception as e:
        print(f"오류 발생: {e}")
