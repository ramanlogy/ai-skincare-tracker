"""
Face Detection Module
Uses MTCNN (Zhang et al., 2016 - cited in project proposal) for
multi-task cascaded face detection and landmark localisation.
"""

import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# Lazy-load MTCNN to avoid slow startup
_mtcnn = None


def _get_mtcnn():
    global _mtcnn
    if _mtcnn is None:
        try:
            from facenet_pytorch import MTCNN
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            _mtcnn = MTCNN(
                keep_all=False,
                device=device,
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7],
                post_process=False
            )
            logger.info(f"MTCNN loaded on {device}")
        except Exception as e:
            logger.error(f"Failed to load MTCNN: {e}")
            _mtcnn = None
    return _mtcnn


def detect_and_align_face(image_path, target_size=(224, 224)):
    """
    Detects a face in the image and returns an aligned, cropped face region.

    Args:
        image_path: Path to input image
        target_size: Output size tuple (width, height)

    Returns:
        dict with keys:
            success     (bool)
            face_image  (PIL.Image or None) - aligned face crop
            box         (list or None)      - [x1,y1,x2,y2]
            landmarks   (np.array or None)  - 5-point landmarks
            confidence  (float)
            message     (str)
    """
    result = {
        'success':    False,
        'face_image': None,
        'box':        None,
        'landmarks':  None,
        'confidence': 0.0,
        'message':    ''
    }

    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        result['message'] = f'Could not open image: {e}'
        return result

    mtcnn = _get_mtcnn()

    # ── MTCNN path ──────────────────────────────────────────────────────────
    if mtcnn is not None:
        try:
            boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
            if boxes is not None and len(boxes) > 0:
                box        = boxes[0]
                confidence = float(probs[0])
                lm         = landmarks[0] if landmarks is not None else None

                # Add 10 % padding around the face box
                w, h   = img.size
                x1, y1, x2, y2 = box
                pad_x  = (x2 - x1) * 0.1
                pad_y  = (y2 - y1) * 0.1
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)

                face_crop = img.crop((x1, y1, x2, y2)).resize(
                    target_size, Image.LANCZOS
                )

                result.update({
                    'success':    True,
                    'face_image': face_crop,
                    'box':        [float(x1), float(y1), float(x2), float(y2)],
                    'landmarks':  lm.tolist() if lm is not None else None,
                    'confidence': confidence,
                    'message':    f'Face detected (confidence: {confidence:.2f})'
                })
                return result

            result['message'] = 'No face detected in image. Please ensure your face is clearly visible and well-lit.'
            return result

        except Exception as e:
            logger.warning(f"MTCNN detection failed: {e}. Falling back to OpenCV.")

    # ── OpenCV Haar fallback ─────────────────────────────────────────────────
    return _opencv_fallback(img, image_path, target_size, result)


def _opencv_fallback(img, image_path, target_size, result):
    """Haar-cascade fallback when facenet-pytorch is unavailable."""
    try:
        import cv2
        cv_img = cv2.imread(str(image_path))
        gray   = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                              minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            x, y, w, h = faces[0]
            pad        = int(min(w, h) * 0.1)
            ih, iw     = cv_img.shape[:2]
            x1 = max(0, x - pad);  y1 = max(0, y - pad)
            x2 = min(iw, x+w+pad); y2 = min(ih, y+h+pad)

            face_crop = img.crop((x1, y1, x2, y2)).resize(target_size, Image.LANCZOS)
            result.update({
                'success':    True,
                'face_image': face_crop,
                'box':        [float(x1), float(y1), float(x2), float(y2)],
                'confidence': 0.75,
                'message':    'Face detected via OpenCV (fallback)'
            })
        else:
            result['message'] = 'No face detected. Ensure good lighting and a clear frontal view.'
    except Exception as e:
        result['message'] = f'Face detection error: {e}'

    return result


def get_face_regions(face_image_array):
    """
    Divides a face image into anatomical regions of interest (ROIs).
    Returns a dict of region_name -> numpy array crop.
    """
    h, w = face_image_array.shape[:2]

    regions = {
        'forehead':    face_image_array[0:int(h*0.30), int(w*0.2):int(w*0.8)],
        'left_cheek':  face_image_array[int(h*0.35):int(h*0.70), 0:int(w*0.40)],
        'right_cheek': face_image_array[int(h*0.35):int(h*0.70), int(w*0.60):w],
        'nose':        face_image_array[int(h*0.35):int(h*0.65), int(w*0.35):int(w*0.65)],
        'chin':        face_image_array[int(h*0.72):h,            int(w*0.25):int(w*0.75)],
    }
    # Remove empty regions caused by rounding
    return {k: v for k, v in regions.items() if v.size > 0}
