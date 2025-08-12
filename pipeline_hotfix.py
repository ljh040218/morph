import os
import json
import argparse
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k):
        return x

Point = Tuple[float, float]

# --------- 사용자 경로 설정 ---------
BASE_DIR = r"C:\Users\NOW\Desktop\tongue"
IMAGES_DIR = os.path.join(BASE_DIR, "image")
LANDMARKS_DIR = os.path.join(BASE_DIR, "results")
MAPPING_JSON = os.path.join(BASE_DIR, 'phoneme_mapping.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# --------- 반모음 파일 매핑 (원본 보존) ---------
SEMIVOWEL_FILES = {
    'ㅣ_반모음_혀': {'image': 'iiJ.png', 'landmarks': 'iiJ_manual_landmarks.json'},
    'ㅜ_반모음_혀': {'image': 'eui-uuW.png', 'landmarks': 'eui-uuW_manual_landmarks.json'},
    'ㅣ_반모음_입': {'image': 'JY-J.png', 'landmarks': 'JY-J_mouth_landmarks.json'},
    'ㅜ_반모음_입': {'image': 'eui-uuW.png', 'landmarks': 'eui-uuW_mouth_landmarks.json'}
}

# --------- 이중모음/반모음 확장 규칙 (원본 보존) ---------
DIPHTHONG_MAPPING = {
    'ㅕ_혀': ('ㅣ_반모음_혀', 'ㅓ_혀'), 'ㅛ_혀': ('ㅗ_혀','ㅅ_혀'),
    'ㅠ_혀': ('ㅣ_반모음_혀', 'ㅜ_혀'), 'ㅑ_혀': ('ㅣ_반모음_혀', 'ㅏ_혀'),
    'ㅞ_혀': ('ㅜ_반모음_혀', 'ㅔ_혀'), 'ㅟ_혀': ('ㅜ_반모음_혀', 'ㅣ_혀'),
    'ㅝ_혀': ('ㅜ_반모음_혀', 'ㅓ_혀'), 'ㅘ_혀': ('ㅜ_반모음_혀', 'ㅏ_혀'),
    'ㅢ_혀': ('ㅣ_혀'), 'ㅒ_혀': ('ㅣ_반모음_혀', 'ㅐ_혀'),
    'ㅖ_혀': ('ㅣ_반모음_혀', 'ㅔ_혀'), 
    
    'ㅕ_입': ('ㅓ_입'), 'ㅛ_입': ('ㅗ_입'), 
    'ㅠ_입': ('ㅜ_입'), 'ㅑ_입': ( 'ㅏ_입'),
    'ㅞ_입': ('ㅔ_입'), 'ㅟ_입': ('ㅣ_입'),
    'ㅝ_입': ('ㅓ_입'), 'ㅘ_입': ('ㅏ_입'),
    'ㅢ_입': ('ㅣ_입'), 'ㅒ_입': ('ㅐ_입'),
    'ㅖ_입': ('ㅔ_입')
}

# --------- 출력 안전용 영문 치환 (원본 보존) ---------
HANGUL_TO_ENGLISH = {
    'ㅏ': 'a', 'ㅓ': 'eo', 'ㅗ': 'o', 'ㅜ': 'u', 'ㅡ': 'eu', 'ㅣ': 'i',
    'ㅑ': 'ya', 'ㅕ': 'yeo', 'ㅛ': 'yo', 'ㅠ': 'yu', 'ㅒ': 'yae', 'ㅖ': 'ye',
    'ㅘ': 'wa', 'ㅙ': 'wae', 'ㅚ': 'oe', 'ㅝ': 'wo', 'ㅞ': 'we', 'ㅟ': 'wi', 'ㅢ': 'ui',
    'ㅐ': 'ae', 'ㅔ': 'e', 'ㄱ': 'g', 'ㄴ': 'n', 'ㄷ': 'd', 'ㄹ': 'r', 'ㅁ': 'm', 
    'ㅂ': 'b', 'ㅅ': 's', 'ㅇ': 'ng', 'ㅈ': 'j', 'ㅊ': 'ch', 'ㅋ': 'k', 'ㅌ': 't', 
    'ㅍ': 'p', 'ㅎ': 'h', 'ㄲ': 'gg', 'ㄸ': 'dd', 'ㅃ': 'bb', 'ㅆ': 'ss', 'ㅉ': 'jj',
    '_입': '_mouth', '_혀': '_tongue', '_반모음': '_semivowel'
}

CONSONANTS = {'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'}
VOWELS = {'ㅏ', 'ㅓ', 'ㅗ', 'ㅜ', 'ㅡ', 'ㅣ', 'ㅐ', 'ㅔ'}
LIAISON_CONSONANTS = {'ㄴ', 'ㄹ', 'ㅁ', 'ㅇ'}

# --------- 유틸 ---------
def safe_convert_hangul(text: str) -> str:
    for hangul, english in HANGUL_TO_ENGLISH.items():
        text = text.replace(hangul, english)
    return text

def read_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def read_points_from_json(path: str) -> List[Point]:
    data = read_json(path)
    pts = [(float(p['x']), float(p['y'])) for p in data]
    return pts

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def create_semivowel_entry(semivowel_type: str, target_type: str) -> Dict:
    if semivowel_type not in SEMIVOWEL_FILES:
        raise KeyError(f"Semivowel type {semivowel_type} not found")
    files = SEMIVOWEL_FILES[semivowel_type]
    return {'label': f"{semivowel_type}_{target_type}", 'image': files['image'], 'landmarks': files['landmarks']}

def expand_diphthong(phoneme: str) -> List[str]:
    if phoneme in DIPHTHONG_MAPPING:
        mapping_value = DIPHTHONG_MAPPING[phoneme]
        target_type = phoneme.split('_')[-1]
        if not isinstance(mapping_value, (list, tuple)):
            mapping_value = (mapping_value,)
        if len(mapping_value) == 3:
            start, semivowel_type, end = mapping_value
            return [start, f"{semivowel_type}_{target_type}", end]
        elif len(mapping_value) == 2:
            start, end = mapping_value
            return [start, end]
        elif len(mapping_value) == 1:
            return [mapping_value[0]]
    return [phoneme]

def expand_phoneme_sequence(phoneme_sequence: List[str]) -> List[str]:
    out = []
    for ph in phoneme_sequence:
        out.extend(expand_diphthong(ph))
    return out

def clamp_points_to_image(points: List[Point], img_shape) -> List[Point]:
    h, w = img_shape[:2]
    return [(max(1, min(w-2, float(x))), max(1, min(h-2, float(y)))) for x, y in points]

def has_self_intersection(poly: List[Point]) -> bool:
    if len(poly) < 4:
        return False
    def seg_inter(a,b,c,d):
        def ccw(p,q,r):
            return (r[1]-p[1])*(q[0]-p[0]) - (q[1]-p[1])*(r[0]-p[0])
        return (ccw(a,b,c)*ccw(a,b,d) < 0) and (ccw(c,d,a)*ccw(c,d,b) < 0)
    for i in range(len(poly)-1):
        for j in range(i+2, len(poly)-1):
            if i == 0 and j == len(poly)-2:
                continue
            if seg_inter(poly[i], poly[i+1], poly[j], poly[j+1]):
                return True
    return False

def add_boundary_points(points: List[Point], img_shape, is_tongue=False) -> List[Point]:
    h, w = img_shape[:2]
    enhanced = points.copy()
    edge_density = 28 if is_tongue else 18
    for i in range(1, edge_density):
        x = (w-1) * i / edge_density
        enhanced.extend([(x, 1), (x, h-2)])
    for i in range(1, edge_density):
        y = (h-1) * i / edge_density
        enhanced.extend([(1, y), (w-2, y)])
    enhanced.extend([(1,1),(w-2,1),(w-2,h-2),(1,h-2)])
    return enhanced

# --------- 색상/밝기 유틸 ---------
COLOR_EMA = {'tongue': None, 'mouth': None}

def mean_std_in_mask(img: np.ndarray, mask: np.ndarray):
    msk = mask > 0
    m = []; s = []
    for c in range(3):
        arr = img[:,:,c][msk]
        if arr.size < 10:
            m.append(0.0); s.append(1.0)
        else:
            m.append(float(arr.mean())); s.append(float(arr.std()+1e-6))
    return np.array(m, dtype=np.float32), np.array(s, dtype=np.float32)

def match_color_to_target(img: np.ndarray, mask: np.ndarray, target_mean: np.ndarray, target_std: np.ndarray):
    out = img.astype(np.float32).copy()
    m, s = mean_std_in_mask(img, mask)
    for c in range(3):
        out[:,:,c] = (out[:,:,c] - m[c]) * (target_std[c] / s[c]) + target_mean[c]
    return np.clip(out, 0, 255).astype(np.uint8)

# --------- 삼각분할 계산 및 캐시 ---------
class TriangulationCache:
    def __init__(self):
        self.cache = {}  # key: (w,h,n_points, is_tongue) -> List[Tuple[int,int,int]]

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, tri):
        self.cache[key] = tri

TRI_CACHE = TriangulationCache()

def calculate_delaunay_triangles(rect, points: List[Point]):
    w, h = rect[2], rect[3]
    safe_points = [(max(1, min(w-2, x)), max(1, min(h-2, y))) for x, y in points]
    subdiv = cv2.Subdiv2D(rect)
    for p in safe_points:
        try:
            subdiv.insert((p[0], p[1]))
        except cv2.error:
            continue
    try:
        triangleList = subdiv.getTriangleList()
    except cv2.error:
        return []
    pts = np.array(safe_points, dtype=np.float32)
    tri_indices = []
    def find_index(pt):
        d = np.linalg.norm(pts - np.array(pt, dtype=np.float32), axis=1)
        idx = int(np.argmin(d))
        return idx if d[idx] < 3.0 else None
    for t in triangleList:
        tp = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        if all(0 <= p[0] < w and 0 <= p[1] < h for p in tp):
            idxs = [find_index(p) for p in tp]
            if None not in idxs and len(set(idxs)) == 3:
                tri_indices.append(tuple(idxs))
    return list({tuple(sorted(tr)): tr for tr in tri_indices}.values())

def apply_affine_transform(src, src_tri, dst_tri, size):
    src_tri = np.array(src_tri, dtype=np.float32)
    dst_tri = np.array(dst_tri, dtype=np.float32)
    warp_mat = cv2.getAffineTransform(src_tri, dst_tri)
    return cv2.warpAffine(src, warp_mat, (size[0], size[1]), None,
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def warp_triangle(img_src, img_dst, t_src, t_dst):
    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))
    if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
        return

    H, W = img_dst.shape[:2]
    x2, y2, w2, h2 = r2
    x2c0 = max(0, x2); y2c0 = max(0, y2)
    x2c1 = min(W, x2 + w2); y2c1 = min(H, y2 + h2)
    w2c = x2c1 - x2c0; h2c = y2c1 - y2c0
    if w2c <= 0 or h2c <= 0:
        return

    x1, y1, w1, h1 = r1
    img1_rect = img_src[y1:y1+h1, x1:x1+w1]

    t1_rect = [((t_src[i][0] - x1), (t_src[i][1] - y1)) for i in range(3)]
    t2_rect = [((t_dst[i][0] - x2), (t_dst[i][1] - y2)) for i in range(3)]

    M = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
    warped_full = cv2.warpAffine(img1_rect, M, (w2, h2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    mask_full = np.zeros((h2, w2, 3), dtype=np.float32)
    cv2.fillConvexPoly(mask_full, np.int32([[(t2_rect[i][0], t2_rect[i][1]) for i in range(3)]]), (1.0, 1.0, 1.0), lineType=cv2.LINE_AA)

    dx0 = x2c0 - x2; dy0 = y2c0 - y2
    dx1 = dx0 + w2c; dy1 = dy0 + h2c

    patch = warped_full[dy0:dy1, dx0:dx1]
    mpatch = mask_full[dy0:dy1, dx0:dx1]

    dst_roi = img_dst[y2c0:y2c1, x2c0:x2c1]
    h = min(dst_roi.shape[0], patch.shape[0]); w = min(dst_roi.shape[1], patch.shape[1])
    dst_roi = dst_roi[:h, :w]; patch = patch[:h, :w]; mpatch = mpatch[:h, :w]

    img_dst[y2c0:y2c1, x2c0:x2c1] = dst_roi * (1 - mpatch) + patch * mpatch

def landmark_mask(points: List[Point], shape, feather: int = 8) -> np.ndarray:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(np.array(points, dtype=np.float32))
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0,0), feather)
    return mask

def enhance_landmarks_with_boundary(pts1: List[Point], pts2: List[Point], img_shape, is_tongue=False):
    pts1 = clamp_points_to_image(pts1, img_shape)
    pts2 = clamp_points_to_image(pts2, img_shape)
    if len(pts1) != len(pts2):
        raise ValueError(f"Landmark counts don't match: {len(pts1)} vs {len(pts2)}")
    if has_self_intersection(pts1[:min(40, len(pts1))]) or has_self_intersection(pts2[:min(40, len(pts2))]):
        print("[warn] self-intersection suspected in landmarks. Check Posterior ordering.")
    enhanced1 = add_boundary_points(pts1, img_shape, is_tongue)
    enhanced2 = pts2.copy(); enhanced2.extend(enhanced1[len(pts1):])
    mask = landmark_mask(pts1, img_shape, feather=10 if is_tongue else 8)
    return clamp_points_to_image(enhanced1, img_shape), clamp_points_to_image(enhanced2, img_shape), mask

def smoothstep(t: float) -> float:
    return t * t * (3 - 2 * t)

def get_interpolation_factor(j: int, frames: int, transition_type: str) -> float:
    t_raw = j / float(frames)
    t_smooth = smoothstep(t_raw)
    if transition_type == 'semivowel':
        return smoothstep(t_smooth)
    elif transition_type == 'consonant_to_vowel':
        return smoothstep(t_smooth ** 1.2)
    elif transition_type == 'vowel_to_consonant':
        return smoothstep(t_smooth ** 1.3)
    else:
        return t_smooth

def morph_two_images(img1, img2, pts1: List[Point], pts2: List[Point], t: float, is_tongue=False,
                     tri_key: Optional[tuple]=None, prev_mid: Optional[List[Point]]=None, prev_frame: Optional[np.ndarray]=None,
                     color_mode: str = 'off', denoise: float = 0.0, sharpen: float = 0.0):
    if img1.shape != img2.shape:
        raise ValueError('Images must be same shape.')
    base_mask = landmark_mask(pts1, img1.shape, feather=6 if is_tongue else 5)

    if color_mode == 'off':
        img1_adj, img2_adj = img1, img2
    elif color_mode == 'roi':
        ref_mean, ref_std = mean_std_in_mask(img1, base_mask)
        to_mean, to_std = mean_std_in_mask(img2, base_mask)
        target_mean = (1.0 - t) * ref_mean + t * to_mean
        target_std  = (1.0 - t) * ref_std  + t * to_std
        img1_adj = match_color_to_target(img1, base_mask, target_mean, target_std)
        img2_adj = match_color_to_target(img2, base_mask, target_mean, target_std)
    else:  # ema
        ref_mean, ref_std = mean_std_in_mask(img1, base_mask)
        to_mean, to_std = mean_std_in_mask(img2, base_mask)
        target_mean = (1.0 - t) * ref_mean + t * to_mean
        target_std  = (1.0 - t) * ref_std  + t * to_std
        key = 'tongue' if is_tongue else 'mouth'
        prev = COLOR_EMA.get(key)
        if prev is None:
            COLOR_EMA[key] = (target_mean, target_std)
        else:
            pm, ps = prev; alpha = 0.12
            COLOR_EMA[key] = ((1-alpha)*pm + alpha*target_mean, (1-alpha)*ps + alpha*target_std)
        em_mean, em_std = COLOR_EMA[key]
        img1_adj = match_color_to_target(img1, base_mask, em_mean, em_std)
        img2_adj = match_color_to_target(img2, base_mask, em_mean, em_std)

    img1f, img2f = img1_adj.astype(np.float32), img2_adj.astype(np.float32)
    try:
        enhanced_pts1, enhanced_pts2, soft_mask = enhance_landmarks_with_boundary(pts1, pts2, img1.shape, is_tongue)
        points_mid = [((1-t)*x1 + t*x2, (1-t)*y1 + t*y2) for (x1,y1),(x2,y2) in zip(enhanced_pts1, enhanced_pts2)]
        points_mid = clamp_points_to_image(points_mid, img1.shape)
        if prev_mid is not None and len(prev_mid) == len(points_mid):
            a = 0.2
            points_mid = [((1-a)*m + a*p, (1-a)*n + a*q) for (m,n),(p,q) in zip(prev_mid, points_mid)]

        w, h = img1.shape[1], img1.shape[0]
        if tri_key is None:
            tri_key = (w, h, len(points_mid), bool(is_tongue))
        tri_idxs = TRI_CACHE.get(tri_key)
        if tri_idxs is None:
            tri_idxs = calculate_delaunay_triangles((0, 0, w, h), points_mid)
            TRI_CACHE.set(tri_key, tri_idxs)
        if not tri_idxs:
            return img1.astype(np.uint8), points_mid

        img1_warped = np.zeros_like(img1f)
        img2_warped = np.zeros_like(img2f)
        for x, y, z in tri_idxs:
            t1 = [enhanced_pts1[x], enhanced_pts1[y], enhanced_pts1[z]]
            t2 = [enhanced_pts2[x], enhanced_pts2[y], enhanced_pts2[z]]
            t_mid = [points_mid[x], points_mid[y], points_mid[z]]
            warp_triangle(img1f, img1_warped, t1, t_mid)
            warp_triangle(img2f, img2_warped, t2, t_mid)

        blended = (1.0 - t) * img1_warped + t * img2_warped
        mask3 = (soft_mask.astype(np.float32) / 255.0)[:, :, None]
        result = blended * mask3 + ((1 - mask3) * ((1.0 - t)*img1f + t*img2f))

        result8 = np.clip(result, 0, 255).astype(np.uint8)
        if prev_frame is not None and prev_frame.shape == result8.shape:
            result8 = cv2.addWeighted(prev_frame, 0.06, result8, 0.94, 0)

        if denoise > 0.0:
            sigma = max(5, int(denoise*10))
            result8 = cv2.bilateralFilter(result8, 5, 20*denoise, 20*denoise)
        if sharpen > 0.0:
            k = np.array([[0,-1,0],[-1,5+sharpen*2,-1],[0,-1,0]], dtype=np.float32)
            result8 = cv2.filter2D(result8, -1, k)

        return result8, points_mid
    except Exception as e:
        print(f"Morphing error: {e}")
        return img1.astype(np.uint8), None

def load_phoneme_map(mapping_path=MAPPING_JSON) -> Dict:
    return read_json(mapping_path)

def load_image_for_entry(entry: Dict):
    img_path = os.path.join(IMAGES_DIR, entry['image'])
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'Image not found: {img_path}')
    return img

def load_landmarks_for_entry(entry: Dict):
    lm_path = os.path.join(LANDMARKS_DIR, entry['landmarks'])
    return read_points_from_json(lm_path)

def assemble_entry(label_key: str, mapping: Dict) -> Dict:
    entry = mapping['phoneme_mapping'].get(label_key)
    if not entry and '_반모음_' in label_key:
        parts = label_key.split('_')
        if len(parts) >= 3:
            try:
                return create_semivowel_entry(f"{parts[0]}_반모음_{parts[2]}", parts[2])
            except KeyError:
                pass
    if not entry:
        raise KeyError(f'Label not found: {label_key}')
    entry = entry.copy()
    entry['label'] = label_key
    return entry

def create_video(frames_dir: str, output_path: str, total_frames: int, fps: int, img_shape, container: str='mp4', archive_mjpg: bool=False):
    h, w = img_shape[:2]
    frames = []
    for frame_idx in range(total_frames):
        fp = os.path.join(frames_dir, f'frame_{frame_idx:04d}.png')
        if os.path.exists(fp):
            fr = cv2.imread(fp)
            if fr is None:
                continue
            if fr.shape[:2] != (h, w):
                fr = cv2.resize(fr, (w, h))
            frames.append(fr)
    if not frames:
        print("Warning: no frames to encode"); 
        return

    mp4_path = output_path if container == 'mp4' else output_path.replace('.avi', '.mp4')
    tried = False
    for fourcc_name in ['avc1', 'H264', 'X264', 'MP4V', 'mp4v']:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        vw = cv2.VideoWriter(mp4_path, fourcc, fps, (w, h))
        if vw.isOpened():
            for fr in frames:
                vw.write(fr)
            vw.release()
            print(f"Video saved: {mp4_path} (codec={fourcc_name}, {len(frames)} frames)")
            tried = True
            break
    if not tried:
        print("Warning: MP4 encoder not available in your OpenCV build.")

    if archive_mjpg:
        avi_path = output_path.replace('.mp4', '.avi')
        for fourcc_name in ['MJPG', 'XVID']:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
            vw = cv2.VideoWriter(avi_path, fourcc, fps, (w, h))
            if vw.isOpened():
                for fr in frames:
                    vw.write(fr)
                vw.release()
                print(f"Archive saved: {avi_path} (codec={fourcc_name}, {len(frames)} frames)")
                break

def get_frame_count(fr_from: str, fr_to: str, i: int, total_entries: int, diphthong_frames: int, 
                    liaison_frames: int, final_frames: int, frames_per_transition: int) -> int:
    from_phoneme = fr_from.split('_')[0]
    to_phoneme = fr_to.split('_')[0]
    is_final_ng = to_phoneme == 'ㅇ' and i < total_entries - 2
    is_middle_ng_transition = from_phoneme == 'ㅇ' and to_phoneme in CONSONANTS
    is_liaison = liaison_frames > 0 and from_phoneme == to_phoneme and from_phoneme in LIAISON_CONSONANTS
    is_semivowel = '반모음' in fr_from or ('반모음' in fr_to and to_phoneme in VOWELS)
    is_consonant_to_vowel = from_phoneme in CONSONANTS and to_phoneme in VOWELS and not is_liaison
    is_vowel_to_consonant = from_phoneme in VOWELS and to_phoneme in CONSONANTS and not is_liaison
    is_diphthong = '반모음' in fr_from or '반모음' in fr_to
    if is_final_ng:
        return max(3, final_frames // 3)
    elif is_middle_ng_transition:
        return max(3, diphthong_frames - 2)
    elif is_liaison:
        return max(3, liaison_frames // 2)
    elif is_semivowel:
        return max(2, diphthong_frames - 3)
    elif is_consonant_to_vowel:
        return max(3, diphthong_frames - 2)
    elif is_vowel_to_consonant:
        return max(3, diphthong_frames - 2)
    elif is_diphthong:
        return max(2, diphthong_frames - 2)
    else:
        return max(4, frames_per_transition // 2)

def generate_transition(fr_from: Dict, fr_to: Dict, frames: int = 10, fps: int = 10, out_root=OUTPUT_DIR, 
                        color_mode: str='off', denoise: float=0.0, sharpen: float=0.0):
    ensure_dir(out_root)
    is_tongue = '_혀' in fr_from.get('label', '')
    from_safe = safe_convert_hangul(fr_from['label'])
    to_safe = safe_convert_hangul(fr_to['label'])
    out_dir = os.path.join(out_root, f"{from_safe}_to_{to_safe}")
    ensure_dir(out_dir)

    img1, img2 = load_image_for_entry(fr_from), load_image_for_entry(fr_to)
    pts1, pts2 = load_landmarks_for_entry(fr_from), load_landmarks_for_entry(fr_to)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    tri_key = None
    prev_mid = None
    prev_frame = None
    for i in tqdm(range(frames + 1)):
        t = i / float(frames)
        frame, prev_mid = morph_two_images(img1, img2, pts1, pts2, t, is_tongue, tri_key, prev_mid, prev_frame, color_mode, denoise, sharpen)
        prev_frame = frame.copy()
        if tri_key is None:
            tri_key = (img1.shape[1], img1.shape[0], len(pts1) + (18*4+4 if is_tongue else 18*4+4), bool(is_tongue))
        cv2.imwrite(os.path.join(out_dir, f'frame_{i:04d}.png'), frame)

    create_video(out_dir, os.path.join(out_root, f"{from_safe}_to_{to_safe}.mp4"), frames + 1, fps, img1.shape)

def generate_sequence(phoneme_sequence: List[str], frames_per_transition: int = 20, diphthong_frames: int = 10,
                      liaison_frames: int = 8, final_frames: int = 8, fps: int = 12, out_root=OUTPUT_DIR,
                      speed_scale: float = 1.0, container: str = 'mp4', archive_mjpg: bool = False,
                      color_mode: str='off', denoise: float=0.0, sharpen: float=0.0):
    ensure_dir(out_root)
    mapping = load_phoneme_map()
    expanded_sequence = expand_phoneme_sequence(phoneme_sequence)

    entries = []
    for phoneme in expanded_sequence:
        try:
            entries.append(assemble_entry(phoneme, mapping))
        except KeyError as e:
            print(f"Error: {e}")
            return

    sequence_name = "_".join(safe_convert_hangul(p) for p in phoneme_sequence)
    out_dir = os.path.join(out_root, f"sequence_{sequence_name}")
    ensure_dir(out_dir)

    total_frame_count = 0
    tri_key = None
    prev_mid = None
    prev_frame = None

    for i in range(len(entries) - 1):
        fr_from, fr_to = entries[i], entries[i + 1]
        if 'ㅇ_' in fr_from.get('label', '') and i == 0:
            continue

        current_frames = get_frame_count(fr_from.get('label', ''), fr_to.get('label', ''), i, len(entries),
                                         diphthong_frames, liaison_frames, final_frames, frames_per_transition)

        img1, img2 = load_image_for_entry(fr_from), load_image_for_entry(fr_to)
        pts1, pts2 = load_landmarks_for_entry(fr_from), load_landmarks_for_entry(fr_to)
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        is_tongue = '_혀' in fr_from.get('label', '')
        from_phoneme = fr_from.get('label', '').split('_')[0]
        to_phoneme = fr_to.get('label', '').split('_')[0]

        transition_type = 'normal'
        if '반모음' in fr_from.get('label', '') or ('반모음' in fr_to.get('label', '') and to_phoneme in VOWELS):
            transition_type = 'semivowel'
        elif from_phoneme in CONSONANTS and to_phoneme in VOWELS:
            transition_type = 'consonant_to_vowel'
        elif from_phoneme in VOWELS and to_phoneme in CONSONANTS:
            transition_type = 'vowel_to_consonant'

        for j in tqdm(range(current_frames + 1), desc=f"Transition {i+1} ({fr_from['label']} → {fr_to['label']})"):
            if i > 0 and j == 0:
                continue
            t = get_interpolation_factor(j, current_frames, transition_type)
            frame, prev_mid = morph_two_images(img1, img2, pts1, pts2, t, is_tongue, tri_key, prev_mid, prev_frame, color_mode, denoise, sharpen)
            prev_frame = frame.copy()
            if tri_key is None:
                tri_key = (img1.shape[1], img1.shape[0], len(pts1) + (18*4+4 if is_tongue else 18*4+4), bool(is_tongue))
            cv2.imwrite(os.path.join(out_dir, f'frame_{total_frame_count:04d}.png'), frame)
            total_frame_count += 1

    if entries:
        last_entry = entries[-1]
        img_last = load_image_for_entry(last_entry)
        hold_frames = max(8, frames_per_transition // 2)
        for _ in tqdm(range(hold_frames), desc="Holding final phoneme"):
            cv2.imwrite(os.path.join(out_dir, f'frame_{total_frame_count:04d}.png'), img_last)
            total_frame_count += 1
        adj_fps = max(1, int(round(fps * speed_scale)))
        create_video(out_dir, os.path.join(out_root, f"sequence_{sequence_name}.mp4"), total_frame_count, adj_fps, img_last.shape, container=container, archive_mjpg=archive_mjpg)

def main_cli():
    parser = argparse.ArgumentParser(description='Korean phoneme morphing (clean)')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    single_parser = subparsers.add_parser('single', help='Generate single transition')
    single_parser.add_argument('--from', dest='frm', type=str, required=True, help='source key')
    single_parser.add_argument('--to', dest='to', type=str, required=True, help='target key')
    single_parser.add_argument('--frames', type=int, default=30, help='number of frames')
    single_parser.add_argument('--fps', type=int, default=10, help='frames per second')
    single_parser.add_argument('--color-mode', type=str, default='off', choices=['off','roi','ema'])
    single_parser.add_argument('--denoise', type=float, default=0.0)
    single_parser.add_argument('--sharpen', type=float, default=0.0)

    sequence_parser = subparsers.add_parser('sequence', help='Generate phoneme sequence')
    sequence_parser.add_argument('--phonemes', type=str, required=True, help='comma-separated phoneme sequence')
    sequence_parser.add_argument('--frames', type=int, default=20, help='frames per transition (default: 20)')
    sequence_parser.add_argument('--diphthong-frames', type=int, default=10, help='frames for diphthong transitions (default: 10)')
    sequence_parser.add_argument('--liaison-frames', type=int, default=8, help='frames for liaison transitions (default: 8)')
    sequence_parser.add_argument('--final-frames', type=int, default=8, help='frames for final consonant transitions (default: 8)')
    sequence_parser.add_argument('--fps', type=int, default=12, help='frames per second (default: 12)')
    sequence_parser.add_argument('--speed-scale', type=float, default=0.9, help='playback speed multiplier (e.g., 0.9 = slightly slower)')
    sequence_parser.add_argument('--container', type=str, default='mp4', choices=['mp4','avi'], help='output container format (default: mp4)')
    sequence_parser.add_argument('--archive-mjpg', action='store_true', help='also save a high-quality MJPG .avi archive')
    sequence_parser.add_argument('--color-mode', type=str, default='off', choices=['off','roi','ema'])
    sequence_parser.add_argument('--denoise', type=float, default=0.0)
    sequence_parser.add_argument('--sharpen', type=float, default=0.0)

    args = parser.parse_args()
    mapping = load_phoneme_map()

    if args.command == 'single':
        fr_entry = assemble_entry(args.frm, mapping)
        to_entry = assemble_entry(args.to, mapping)
        generate_transition(fr_entry, to_entry, frames=args.frames, fps=args.fps, color_mode=args.color_mode, denoise=args.denoise, sharpen=args.sharpen)
    elif args.command == 'sequence':
        phoneme_list = [p.strip() for p in args.phonemes.split(',')]
        generate_sequence(phoneme_list, frames_per_transition=args.frames, 
                          diphthong_frames=getattr(args, 'diphthong_frames', 10),
                          liaison_frames=getattr(args, 'liaison_frames', 8),
                          final_frames=getattr(args, 'final_frames', 8), 
                          fps=args.fps,
                          speed_scale=args.speed_scale,
                          container=args.container,
                          archive_mjpg=args.archive_mjpg,
                          color_mode=args.color_mode,
                          denoise=args.denoise,
                          sharpen=args.sharpen)
    else:
        parser.print_help()

if __name__ == '__main__':
    main_cli()
