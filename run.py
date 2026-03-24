#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║   SanjuAI — AI Skincare Progress Tracker                         ║
║   COM668 Computing Project · Student ID: B00912171              ║
║   Project Title: AI Skincare Progress Tracker                   ║
║                                                                  ║
║   SINGLE FILE — run with:  python run.py                        ║
║   Then open:               http://127.0.0.1:5000                ║
╚══════════════════════════════════════════════════════════════════╝

Install dependencies first (one-time):
    pip install flask flask-sqlalchemy flask-login werkzeug \
                opencv-python-headless Pillow numpy scipy \
                scikit-image scikit-learn torch torchvision \
                facenet-pytorch matplotlib pandas
"""

# ═══════════════════════════════════════════════════════════════════
# IMPORTS & SETUP
# ═══════════════════════════════════════════════════════════════════

import os, uuid, logging
import numpy as np
import cv2
from PIL import Image
from datetime import datetime

BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
DB_PATH    = os.path.join(BASE_DIR, 'skincare.db')
os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Sanjuai')


# ═══════════════════════════════════════════════════════════════════
# SECTION 1 — FACE DETECTION  (Zhang et al., 2016 — MTCNN)
# ═══════════════════════════════════════════════════════════════════

_mtcnn = None

def _get_mtcnn():
    global _mtcnn
    if _mtcnn is None:
        try:
            from facenet_pytorch import MTCNN
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            _mtcnn = MTCNN(keep_all=False, device=device,
                           min_face_size=40, thresholds=[0.6, 0.7, 0.7],
                           post_process=False)
            logger.info(f"MTCNN loaded on {device}")
        except Exception as e:
            logger.warning(f"MTCNN unavailable ({e}), will use OpenCV fallback")
    return _mtcnn


def detect_and_align_face(image_path, target_size=(224, 224)):
    """Detect face using MTCNN (Zhang et al.) with OpenCV Haar fallback."""
    result = dict(success=False, face_image=None, box=None,
                  landmarks=None, confidence=0.0, message='')
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        result['message'] = f'Cannot open image: {e}'
        return result

    mtcnn = _get_mtcnn()
    if mtcnn is not None:
        try:
            boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
            if boxes is not None and len(boxes):
                conf = float(probs[0])
                x1, y1, x2, y2 = boxes[0]
                w, h = img.size
                px, py = (x2 - x1) * .10, (y2 - y1) * .10
                x1 = max(0, x1 - px); y1 = max(0, y1 - py)
                x2 = min(w, x2 + px); y2 = min(h, y2 + py)
                face = img.crop((x1, y1, x2, y2)).resize(target_size, Image.LANCZOS)
                result.update(success=True, face_image=face,
                              box=[float(x1), float(y1), float(x2), float(y2)],
                              landmarks=landmarks[0].tolist() if landmarks is not None else None,
                              confidence=conf,
                              message=f'Face detected (confidence {conf:.2f})')
                return result
            result['message'] = 'No face found — ensure good lighting and a clear frontal view.'
            return result
        except Exception as e:
            logger.warning(f"MTCNN failed: {e}, trying OpenCV Haar cascade")

    # ── OpenCV Haar Cascade fallback ────────────────────────────────
    try:
        cv_img = cv2.imread(str(image_path))
        if cv_img is None:
            result['message'] = 'Could not read image file.'
            return result
        gray  = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        casc  = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = casc.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if len(faces):
            x, y, fw, fh = faces[0]
            pad = int(min(fw, fh) * .1)
            ih, iw = cv_img.shape[:2]
            x1 = max(0, x - pad);       y1 = max(0, y - pad)
            x2 = min(iw, x + fw + pad); y2 = min(ih, y + fh + pad)
            face = img.crop((x1, y1, x2, y2)).resize(target_size, Image.LANCZOS)
            result.update(success=True, face_image=face,
                          box=[float(x1), float(y1), float(x2), float(y2)],
                          confidence=0.75,
                          message='Face detected via OpenCV cascade')
        else:
            result['message'] = ('No face detected. Ensure good natural lighting '
                                 'and a clear frontal view.')
    except Exception as e:
        result['message'] = f'Detection error: {e}'
    return result


def get_face_regions(face_arr):
    """Split face into anatomical ROIs for per-region analysis."""
    h, w = face_arr.shape[:2]
    regions = dict(
        forehead    = face_arr[0:int(h * .30), int(w * .2):int(w * .8)],
        left_cheek  = face_arr[int(h * .35):int(h * .70), 0:int(w * .40)],
        right_cheek = face_arr[int(h * .35):int(h * .70), int(w * .60):w],
        nose        = face_arr[int(h * .35):int(h * .65), int(w * .35):int(w * .65)],
        chin        = face_arr[int(h * .72):h,             int(w * .25):int(w * .75)],
    )
    return {k: v for k, v in regions.items() if v.size > 0}


# ═══════════════════════════════════════════════════════════════════
# SECTION 2 — PREPROCESSING  (CLAHE + Grey-World)
# ═══════════════════════════════════════════════════════════════════

def _grey_world(arr):
    img  = arr.astype(np.float32)
    mean = img.mean()
    for c in range(3):
        cm = img[:, :, c].mean()
        img[:, :, c] *= mean / (cm + 1e-6)
    return np.clip(img, 0, 255).astype(np.uint8)


def _clahe(arr):
    """CLAHE on L-channel of LAB — supervisor's recommended approach."""
    bgr      = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    lab      = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b  = cv2.split(lab)
    l        = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    bgr2     = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)


def preprocess_face(face_image, target_size=(224, 224)):
    if isinstance(face_image, Image.Image):
        arr = np.array(face_image.convert('RGB').resize(target_size, Image.LANCZOS))
    else:
        arr = cv2.resize(face_image, target_size)
    arr = _grey_world(arr)
    arr = _clahe(arr)
    arr = cv2.GaussianBlur(arr, (3, 3), sigmaX=0.5)
    return arr


# ═══════════════════════════════════════════════════════════════════
# SECTION 3 — ACNE DETECTION  (Blob detection + HSV)
# ═══════════════════════════════════════════════════════════════════

def detect_acne(face_arr, regions=None):
    """Approach 1 MVP: blob detection + HSV thresholding (supervisor rec.)"""
    if face_arr is None or face_arr.size == 0:
        return dict(acne_count=0.0, acne_score=50.0,
                    spot_mask=None, annotated_image=None)

    bgr = cv2.cvtColor(face_arr, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Red hue — wraps at 0/180 in OpenCV
    m1 = cv2.inRange(hsv, np.array([0,   50,  50]), np.array([10,  255, 255]))
    m2 = cv2.inRange(hsv, np.array([160, 50,  50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(m1, m2)

    # Dark-spot mask (comedones / pores)
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, dm = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    combined = cv2.bitwise_or(red_mask, dm)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k, iterations=1)

    p = cv2.SimpleBlobDetector_Params()
    p.filterByArea       = True;  p.minArea = 20;  p.maxArea = 2000
    p.filterByCircularity = True; p.minCircularity = 0.2
    p.filterByConvexity  = False; p.filterByInertia = False
    kps = cv2.SimpleBlobDetector_create(p).detect(cv2.bitwise_not(combined))

    count = len(kps)
    bp    = np.sum(combined > 0) / (face_arr.shape[0] * face_arr.shape[1] + 1e-6)
    score = round(max(0.0, 100.0 - count * 4) * 0.6 +
                  max(0.0, 100.0 - bp * 500)  * 0.4, 1)

    ann = face_arr.copy()
    for kp in kps:
        cx, cy = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(ann, (cx, cy), max(int(kp.size / 2), 4), (255, 80, 80), 2)

    return dict(acne_count=float(count), acne_score=score,
                spot_mask=combined, annotated_image=ann)


# ═══════════════════════════════════════════════════════════════════
# SECTION 4 — REDNESS ANALYSIS  (HSV hue range)
# ═══════════════════════════════════════════════════════════════════

def analyse_redness(face_arr):
    if face_arr is None or face_arr.size == 0:
        return dict(redness_score=50.0, redness_pct=0.0, redness_map=None)

    bgr = cv2.cvtColor(face_arr, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    m1 = cv2.inRange(hsv, np.array([0,   40, 40]), np.array([12,  255, 255]))
    m2 = cv2.inRange(hsv, np.array([155, 40, 40]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(m1, m2)

    # Exclude dark / shadow pixels
    bright   = (hsv[:, :, 2] > 40).astype(np.uint8) * 255
    red_mask = cv2.bitwise_and(red_mask, bright)

    redness_pct   = np.sum(red_mask > 0) / (np.sum(bright > 0) + 1e-6) * 100
    redness_score = round(max(0.0, 100.0 - (redness_pct / 30.0) * 100.0), 1)

    blurred = cv2.GaussianBlur(
        (red_mask / 255.0).astype(np.float32), (21, 21), 0)
    hmap = cv2.cvtColor(
        cv2.applyColorMap((blurred * 255).astype(np.uint8), cv2.COLORMAP_HOT),
        cv2.COLOR_BGR2RGB)

    return dict(redness_score=redness_score,
                redness_pct=round(float(redness_pct), 2),
                redness_map=hmap)


# ═══════════════════════════════════════════════════════════════════
# SECTION 5 — TEXTURE ANALYSIS  (LBP — Ojala et al., 2002)
# ═══════════════════════════════════════════════════════════════════

def _lbp_numpy(gray, radius=1, n_points=8):
    """Pure-NumPy LBP — no scikit-image required."""
    h, w    = gray.shape
    lbp     = np.zeros((h, w), dtype=np.uint8)
    padded  = np.pad(gray, radius, mode='reflect').astype(np.float32)
    center  = gray.astype(np.float32)
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        dy    =  radius * np.sin(angle)
        dx    = -radius * np.cos(angle)
        yy, xx = np.meshgrid(
            np.arange(radius, radius + h, dtype=np.float32) + dy,
            np.arange(radius, radius + w, dtype=np.float32) + dx,
            indexing='ij')
        y0 = np.clip(np.floor(yy).astype(int), 0, padded.shape[0] - 1)
        x0 = np.clip(np.floor(xx).astype(int), 0, padded.shape[1] - 1)
        y1 = np.clip(y0 + 1, 0, padded.shape[0] - 1)
        x1 = np.clip(x0 + 1, 0, padded.shape[1] - 1)
        fy, fx = yy - np.floor(yy), xx - np.floor(xx)
        interp = (padded[y0, x0] * (1 - fy) * (1 - fx) +
                  padded[y1, x0] *      fy   * (1 - fx) +
                  padded[y0, x1] * (1 - fy)  *      fx  +
                  padded[y1, x1] *      fy   *      fx)
        lbp += ((interp >= center).astype(np.uint8) << i)
    return lbp


def analyse_texture(face_arr):
    if face_arr is None or face_arr.size == 0:
        return dict(texture_score=50.0, uniformity=0.5, lbp_image=None)

    gray = cv2.cvtColor(face_arr, cv2.COLOR_RGB2GRAY)
    try:
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(gray, P=8, R=1,
                                   method='uniform').astype(np.uint8)
    except ImportError:
        lbp = _lbp_numpy(gray)

    hist, _ = np.histogram(lbp.ravel(), bins=256,
                           range=(0, 256), density=True)
    uniformity = float(np.sum(hist ** 2))
    var        = float(np.var(lbp))
    score      = round(max(0.0, 100.0 - (var / 4000.0) * 100.0), 1)

    norm  = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX)
    lbp_c = cv2.cvtColor(
        cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)

    return dict(texture_score=score,
                uniformity=round(uniformity, 4),
                lbp_image=lbp_c)


# ═══════════════════════════════════════════════════════════════════
# SECTION 6 — COMPOSITE SCORER
# ═══════════════════════════════════════════════════════════════════

WEIGHTS = dict(acne=0.4, redness=0.3, texture=0.3)

def composite_score(acne, redness, texture):
    """Supervisor's recommended weighted formula."""
    return round(acne * WEIGHTS['acne'] +
                 redness * WEIGHTS['redness'] +
                 texture * WEIGHTS['texture'], 1)


def compute_change(current, baseline):
    if baseline is None or baseline == 0:
        return 0.0
    return round(current - baseline, 1)


def analyse_image(image_path, baseline_scan=None):
    """Full pipeline: detect → preprocess → analyse → score → compare."""
    r = dict(face_detected=False, acne_count=0.0, acne_score=50.0,
             redness_score=50.0, texture_score=50.0, overall_score=50.0,
             acne_change=0.0, redness_change=0.0, texture_change=0.0,
             overall_change=0.0, analysis_status='failed', message='')

    det = detect_and_align_face(image_path)
    if not det['success']:
        r['message'] = det['message']
        return r

    r['face_detected'] = True
    face_arr = preprocess_face(det['face_image'])
    regions  = get_face_regions(face_arr)

    ar = detect_acne(face_arr, regions)
    rr = analyse_redness(face_arr)
    tr = analyse_texture(face_arr)

    a_s     = ar['acne_score']
    red_s   = rr['redness_score']
    tex_s   = tr['texture_score']
    overall = composite_score(a_s, red_s, tex_s)

    ac = rc = tc = oc = 0.0
    if baseline_scan is not None:
        ac = compute_change(a_s,    baseline_scan.acne_score)
        rc = compute_change(red_s,  baseline_scan.redness_score)
        tc = compute_change(tex_s,  baseline_scan.texture_score)
        oc = compute_change(overall, baseline_scan.overall_score)

    r.update(acne_count=ar['acne_count'], acne_score=a_s,
             redness_score=red_s, texture_score=tex_s,
             overall_score=overall,
             acne_change=ac, redness_change=rc,
             texture_change=tc, overall_change=oc,
             analysis_status='complete', message='Analysis complete.')
    return r


# ═══════════════════════════════════════════════════════════════════
# SECTION 7 — FLASK APP + MODELS
# ═══════════════════════════════════════════════════════════════════

from flask import (Flask, render_template_string, request, redirect,
                   url_for, flash, jsonify, abort, send_from_directory)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin, login_user, logout_user,
                         login_required, current_user)
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config.update(
    SECRET_KEY              = 'Sanjuai-skincare-secret-2026',
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{DB_PATH}',
    SQLALCHEMY_TRACK_MODIFICATIONS = False,
    UPLOAD_FOLDER           = UPLOAD_DIR,
    MAX_CONTENT_LENGTH      = 16 * 1024 * 1024,
)

db            = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view             = 'login_page'
login_manager.login_message          = 'Please log in to continue.'
login_manager.login_message_category = 'info'


@login_manager.user_loader
def load_user(uid):
    return User.query.get(int(uid))


class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80),  unique=True, nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    consent_given = db.Column(db.Boolean,  default=False)
    scans         = db.relationship('SkinScan', backref='user', lazy=True,
                                    cascade='all, delete-orphan')

    def set_password(self, p):
        self.password_hash = generate_password_hash(p)

    def check_password(self, p):
        return check_password_hash(self.password_hash, p)


class SkinScan(db.Model):
    __tablename__   = 'skin_scans'
    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    image_filename  = db.Column(db.String(256), nullable=False)
    captured_at     = db.Column(db.DateTime, default=datetime.utcnow)
    notes           = db.Column(db.Text, default='')
    face_detected   = db.Column(db.Boolean, default=False)
    acne_count      = db.Column(db.Float, default=0.0)
    acne_score      = db.Column(db.Float, default=50.0)
    redness_score   = db.Column(db.Float, default=50.0)
    texture_score   = db.Column(db.Float, default=50.0)
    overall_score   = db.Column(db.Float, default=50.0)
    acne_change     = db.Column(db.Float, default=0.0)
    redness_change  = db.Column(db.Float, default=0.0)
    texture_change  = db.Column(db.Float, default=0.0)
    overall_change  = db.Column(db.Float, default=0.0)
    analysis_status = db.Column(db.String(20), default='pending')

    def to_dict(self):
        return dict(
            id=self.id,
            image_filename=self.image_filename,
            captured_at=self.captured_at.strftime('%Y-%m-%d %H:%M'),
            notes=self.notes,
            face_detected=self.face_detected,
            acne_count=round(self.acne_count, 1),
            acne_score=round(self.acne_score, 1),
            redness_score=round(self.redness_score, 1),
            texture_score=round(self.texture_score, 1),
            overall_score=round(self.overall_score, 1),
            acne_change=round(self.acne_change, 1),
            redness_change=round(self.redness_change, 1),
            texture_change=round(self.texture_change, 1),
            overall_change=round(self.overall_change, 1),
            analysis_status=self.analysis_status,
        )


# ═══════════════════════════════════════════════════════════════════
# SECTION 8 — CSS (shared across all pages)
# ═══════════════════════════════════════════════════════════════════

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;1,300;1,400&family=DM+Sans:wght@300;400;500&display=swap');

:root {
  --cream:      #faf7f4; --parch:  #f2ede6; --ww:      #fffcf9;
  --sage:       #4a7c59; --sage-l: #6fa07a; --sage-d:  #325440;
  --rose:       #c17b6c; --rose-l: #d9a89c; --sky:     #6c9ec1;
  --ink:        #1a1a1a; --ink-l:  #4a4a4a; --ink-m:   #8a8a8a;
  --border:     #e5dfd7; --bd:     #cbc4bb;
  --r: 12px; --rs: 8px; --rl: 20px; --t: .2s ease;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { font-size: 16px; -webkit-font-smoothing: antialiased; }
body { font-family: 'DM Sans', sans-serif; background: var(--cream);
       color: var(--ink); min-height: 100vh; display: flex; flex-direction: column; }
h1,h2,h3,h4 { font-family: 'Cormorant Garamond', serif; font-weight: 500; line-height: 1.2; }
a { color: var(--sage); text-decoration: none; }
a:hover { color: var(--sage-d); }
img { max-width: 100%; display: block; }
main { flex: 1; }

/* ── Navbar ── */
.navbar { background: var(--ww); border-bottom: 1px solid var(--border);
  display: flex; align-items: center; padding: 0 2rem; height: 60px;
  position: sticky; top: 0; z-index: 100; box-shadow: 0 1px 3px rgba(0,0,0,.07); }
.nav-brand { font-family: 'Cormorant Garamond', serif; font-size: 1.4rem;
  font-weight: 500; color: var(--ink); display: flex; align-items: center;
  gap: .4rem; letter-spacing: .02em; }
.nav-brand:hover { color: var(--sage); }
.brand-mark { color: var(--sage); }
.nav-links { display: flex; gap: .25rem; margin-left: 2.5rem; }
.nav-links a { font-size: .85rem; font-weight: 500; color: var(--ink-l);
  padding: .4rem .9rem; border-radius: 6px; transition: var(--t); }
.nav-links a:hover, .nav-links a.active { background: var(--parch); color: var(--sage-d); }
.nav-user { margin-left: auto; display: flex; align-items: center; gap: .75rem; }
.user-chip { font-size: .8rem; background: var(--parch); border: 1px solid var(--border);
  padding: .25rem .75rem; border-radius: 20px; color: var(--ink-l); }
.btn-ghost-sm { font-size: .8rem; color: var(--ink-m); padding: .25rem .6rem;
  border-radius: 6px; border: 1px solid var(--border); transition: var(--t);
  display: inline-block; }
.btn-ghost-sm:hover { background: var(--parch); color: var(--ink); }

/* ── Flash messages ── */
.flash-container { position: fixed; top: 70px; right: 1.5rem; z-index: 200;
  display: flex; flex-direction: column; gap: .5rem; max-width: 380px; }
.flash { padding: .85rem 1.1rem; border-radius: var(--rs); font-size: .875rem;
  display: flex; align-items: flex-start; justify-content: space-between;
  gap: .75rem; box-shadow: 0 4px 16px rgba(0,0,0,.09);
  animation: slideIn .25s ease; line-height: 1.4; }
@keyframes slideIn { from { opacity:0; transform:translateX(20px); }
                     to   { opacity:1; transform:translateX(0); } }
.flash-success { background:#f0f7f2; border:1px solid #b8d9c4; color:#2d6944; }
.flash-error   { background:#fdf2f0; border:1px solid #e8c0b8; color:#8b3a2c; }
.flash-warning { background:#fdf8ed; border:1px solid #e8d8a0; color:#7a5f1a; }
.flash-info    { background:#f0f5fb; border:1px solid #b8cfe8; color:#2d5580; }
.flash-close { background:none; border:none; font-size:1.1rem; cursor:pointer;
  color:inherit; opacity:.5; line-height:1; flex-shrink:0; }
.flash-close:hover { opacity:1; }

/* ── Page layout ── */
.pc { max-width: 1100px; margin: 0 auto; padding: 2.5rem 1.5rem 4rem; }
.pc--n { max-width: 720px; }
.ph { display: flex; align-items: flex-end; justify-content: space-between;
  margin-bottom: 2rem; flex-wrap: wrap; gap: 1rem; }
.pt { font-size: 2rem; letter-spacing: -.01em; }
.ps { font-size: .9rem; color: var(--ink-m); margin-top: .2rem; }

/* ── Buttons ── */
.btn-primary { background: var(--sage); color: #fff; border: none;
  padding: .7rem 1.5rem; border-radius: var(--rs); font-family: 'DM Sans', sans-serif;
  font-size: .875rem; font-weight: 500; cursor: pointer; transition: var(--t);
  letter-spacing: .02em; display: inline-block; }
.btn-primary:hover { background: var(--sage-d); color: #fff; transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(74,124,89,.3); }
.btn-primary.fw { width:100%; text-align:center; padding:.9rem; }
.btn-primary:disabled { background:var(--bd); cursor:not-allowed;
  transform:none; box-shadow:none; }
.btn-ghost { background:transparent; color:var(--ink-l); border:1px solid var(--bd);
  padding:.6rem 1.2rem; border-radius:var(--rs); font-family:'DM Sans',sans-serif;
  font-size:.875rem; cursor:pointer; transition:var(--t); }
.btn-ghost:hover { background:var(--parch); color:var(--ink); }
.btn-danger { background:transparent; color:#c0392b; border:1px solid #e8c0b8;
  padding:.6rem 1.2rem; border-radius:var(--rs); font-family:'DM Sans',sans-serif;
  font-size:.875rem; cursor:pointer; transition:var(--t); }
.btn-danger:hover { background:#fdf2f0; border-color:#c0392b; }
.btn-dsm { background:none; border:1px solid #e8c0b8; color:#c0392b;
  font-size:.78rem; padding:.3rem .7rem; border-radius:5px; cursor:pointer;
  font-family:'DM Sans',sans-serif; transition:var(--t); }
.btn-dsm:hover { background:#fdf2f0; }

/* ── Cards ── */
.card { background:var(--ww); border:1px solid var(--border); border-radius:var(--r);
  padding:1.75rem; box-shadow:0 1px 3px rgba(0,0,0,.07); margin-bottom:1.5rem; }
.ct { font-size:1.1rem; color:var(--ink); margin-bottom:1.25rem;
  padding-bottom:.75rem; border-bottom:1px solid var(--border); }

/* ── Score grid ── */
.sg { display:grid; grid-template-columns:1fr 1fr 1fr 1fr;
  gap:1rem; margin-bottom:1.5rem; }
@media(max-width:900px){ .sg{ grid-template-columns:1fr 1fr; } }
@media(max-width:560px){ .sg{ grid-template-columns:1fr; } }
.sc { background:var(--ww); border:1px solid var(--border); border-radius:var(--r);
  padding:1.25rem 1.5rem; box-shadow:0 1px 3px rgba(0,0,0,.07);
  display:flex; flex-direction:column; gap:.75rem; }
.sc--o { background:var(--sage); border-color:var(--sage-d); color:#fff; align-items:center; }
.sl { font-size:.78rem; font-weight:500; text-transform:uppercase;
  letter-spacing:.08em; color:var(--ink-m); }
.sc--o .sl { color:rgba(255,255,255,.8); }
.sring { position:relative; width:80px; height:80px; }
.rsv { width:80px; height:80px; }
.rv { position:absolute; inset:0; display:flex; align-items:center;
  justify-content:center; font-family:'Cormorant Garamond',serif;
  font-size:1.6rem; font-weight:500; color:#fff; }
.bscore { display:flex; align-items:center; gap:.75rem; }
.btrack { flex:1; height:6px; background:var(--parch); border-radius:3px;
  overflow:hidden; border:1px solid var(--border); }
.bfill { height:100%; border-radius:3px; transition:width .6s ease; }
.bnum { font-family:'Cormorant Garamond',serif; font-size:1.3rem;
  min-width:2.2rem; text-align:right; }
.schg { font-size:.78rem; font-weight:500; }
.schg.pos { color:var(--sage-l); } .schg.neg { color:var(--rose); }
.sc--o .schg.pos { color:#a8e6be; } .sc--o .schg.neg { color:#f7a999; }

/* ── Two-column layout ── */
.sr2 { display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin-bottom:1.5rem; }
@media(max-width:700px){ .sr2{ grid-template-columns:1fr; } }

/* ── Scan preview ── */
.sprev { display:flex; gap:1.25rem; align-items:flex-start; }
.simg  { width:120px; height:120px; object-fit:cover; border-radius:var(--rs);
  border:1px solid var(--border); flex-shrink:0; }
.smeta { font-size:.875rem; line-height:1.8; color:var(--ink-l); }
.smeta strong { color:var(--ink); }

/* ── Compare ── */
.compare-row { display:flex; align-items:center; gap:1rem; justify-content:center; }
.compare-col { text-align:center; }
.compare-label { font-size:.75rem; text-transform:uppercase; letter-spacing:.08em;
  color:var(--ink-m); margin-bottom:.5rem; }
.compare-img { width:100px; height:100px; object-fit:cover;
  border-radius:var(--rs); border:1px solid var(--border); }
.compare-date { font-size:.75rem; color:var(--ink-m); margin-top:.4rem; }
.compare-arrow { font-size:1.5rem; color:var(--sage); }

/* ── Chart tabs ── */
.chtabs { display:flex; gap:.4rem; margin-bottom:1.5rem; flex-wrap:wrap; }
.tb { background:var(--parch); border:1px solid var(--border); color:var(--ink-l);
  padding:.4rem .9rem; border-radius:6px; font-family:'DM Sans',sans-serif;
  font-size:.8rem; font-weight:500; cursor:pointer; transition:var(--t); }
.tb:hover { border-color:var(--sage); color:var(--sage); }
.tb.active { background:var(--sage); border-color:var(--sage); color:#fff; }
.chtwrap canvas { max-height:280px; }

/* ── Badges ── */
.badge { display:inline-block; font-size:.72rem; font-weight:500;
  padding:.2rem .55rem; border-radius:20px; text-transform:uppercase; letter-spacing:.05em; }
.badge--complete { background:#eef7f1; color:#2d6944; border:1px solid #b8d9c4; }
.badge--pending  { background:#fdf8ed; color:#7a5f1a; border:1px solid #e8d8a0; }
.badge--failed   { background:#fdf2f0; color:#8b3a2c; border:1px solid #e8c0b8; }

/* ── Empty state ── */
.empty-state { text-align:center; padding:5rem 1rem; color:var(--ink-m); }
.empty-icon  { font-size:3rem; color:var(--bd); margin-bottom:1rem; }
.empty-state h2 { font-size:1.5rem; color:var(--ink-l); margin-bottom:.5rem; }
.empty-state p  { margin-bottom:1.5rem; font-size:.95rem; }

/* ── Auth layout ── */
.auth-layout { min-height:100vh; display:grid; grid-template-columns:1fr 1fr; }
@media(max-width:768px){ .auth-layout{ grid-template-columns:1fr; } .auth-hero{ display:none; } }
.auth-hero { background:var(--sage-d); display:flex; align-items:center;
  justify-content:center; padding:3rem; position:relative; overflow:hidden; }
.auth-hero::before { content:''; position:absolute; inset:0;
  background:repeating-linear-gradient(45deg,transparent,transparent 40px,
  rgba(255,255,255,.025) 40px,rgba(255,255,255,.025) 80px); }
.hero-inner  { position:relative; z-index:1; color:#fff; max-width:380px; }
.hero-eyebrow { font-size:.75rem; font-weight:500; text-transform:uppercase;
  letter-spacing:.12em; color:rgba(255,255,255,.55); margin-bottom:1rem; }
.hero-title { font-family:'Cormorant Garamond',serif; font-size:4rem; font-weight:300;
  color:#fff; line-height:1; margin-bottom:1rem; }
.hero-title em { font-style:italic; color:#a8e6be; }
.hero-sub  { font-size:.95rem; color:rgba(255,255,255,.75); line-height:1.6; margin-bottom:2.5rem; }
.hero-feats { display:flex; flex-direction:column; gap:.75rem; }
.feat { display:flex; align-items:center; gap:.75rem; font-size:.875rem; color:rgba(255,255,255,.8); }
.feat-icon { color:#a8e6be; font-size:.75rem; }
.auth-forms { display:flex; flex-direction:column; align-items:center;
  justify-content:center; padding:3rem 2rem; background:var(--cream); }
.form-card { background:var(--ww); border:1px solid var(--border);
  border-radius:var(--rl); padding:2.5rem; width:100%; max-width:400px;
  box-shadow:0 4px 16px rgba(0,0,0,.09); }
.form-title { font-size:1.6rem; margin-bottom:.3rem; }
.form-sub   { font-size:.875rem; color:var(--ink-m); margin-bottom:1.75rem; }
.form-switch { font-size:.85rem; color:var(--ink-m); text-align:center; margin-top:1.25rem; }
.form-switch a { color:var(--sage); font-weight:500; }
.disc-mini { font-size:.75rem; color:var(--ink-m); text-align:center;
  max-width:360px; margin-top:1.25rem; line-height:1.5; }

/* ── Form fields ── */
.field { margin-bottom:1rem; }
.field label { display:block; font-size:.8rem; font-weight:500; color:var(--ink-l);
  text-transform:uppercase; letter-spacing:.07em; margin-bottom:.4rem; }
.opt { font-weight:400; text-transform:none; letter-spacing:0;
  color:var(--ink-m); font-size:.75rem; }
.field input, .field textarea { width:100%; padding:.7rem .9rem;
  border:1px solid var(--bd); border-radius:var(--rs); font-family:'DM Sans',sans-serif;
  font-size:.9rem; background:var(--cream); color:var(--ink); transition:var(--t); }
.field input:focus, .field textarea:focus { outline:none; border-color:var(--sage);
  box-shadow:0 0 0 3px rgba(74,124,89,.1); background:var(--ww); }
.consent-box { background:var(--parch); border:1px solid var(--border);
  border-radius:var(--rs); padding:.9rem 1rem; margin-bottom:1.25rem; }
.consent-label { display:flex; gap:.7rem; align-items:flex-start; font-size:.82rem;
  color:var(--ink-l); line-height:1.5; cursor:pointer; }
.consent-label input[type=checkbox] { width:auto; margin-top:.2rem;
  flex-shrink:0; accent-color:var(--sage); }
.consent-reminder { font-size:.78rem; color:var(--ink-m); background:var(--parch);
  border:1px solid var(--border); border-radius:var(--rs);
  padding:.75rem 1rem; margin-bottom:1.25rem; line-height:1.5; }

/* ── Upload zone ── */
.upload-zone { border:2px dashed var(--bd); border-radius:var(--r);
  background:var(--parch); margin-bottom:1.5rem; transition:var(--t);
  min-height:200px; display:flex; align-items:center; justify-content:center; }
.upload-zone:hover, .upload-zone.drag-over { border-color:var(--sage); background:#f0f7f2; }
#uploadPlaceholder { text-align:center; padding:2.5rem; cursor:pointer; width:100%; }
.upload-icon { font-size:2.5rem; color:var(--sage); margin-bottom:.75rem; }
.upload-cta  { font-weight:500; color:var(--ink-l); margin-bottom:.3rem; }
.upload-hint { font-size:.8rem; color:var(--ink-m); }
.preview-image  { width:100%; max-height:320px; object-fit:contain;
  border-radius:var(--rs); padding:.75rem; }
.preview-actions { padding:.75rem; text-align:center; border-top:1px solid var(--border); }

/* ── Tips ── */
.tips-bar { display:flex; gap:.75rem; margin-bottom:1.5rem; flex-wrap:wrap; }
.tip { display:flex; align-items:center; gap:.4rem; background:var(--ww);
  border:1px solid var(--border); border-radius:20px; padding:.35rem .9rem;
  font-size:.8rem; color:var(--ink-l); }

/* ── Info grid ── */
.info-grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:1rem; margin-top:1.5rem; }
@media(max-width:600px){ .info-grid{ grid-template-columns:1fr; } }
.info-card { background:var(--ww); border:1px solid var(--border);
  border-radius:var(--r); padding:1.25rem; }
.info-card h4 { font-size:.95rem; color:var(--sage-d); margin-bottom:.5rem; }
.info-card p  { font-size:.82rem; color:var(--ink-m); line-height:1.6; }

/* ── History grid ── */
.history-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr));
  gap:1rem; margin-bottom:2rem; }
.hcard { background:var(--ww); border:1px solid var(--border); border-radius:var(--r);
  overflow:hidden; box-shadow:0 1px 3px rgba(0,0,0,.07); transition:var(--t); }
.hcard:hover { box-shadow:0 4px 16px rgba(0,0,0,.09); transform:translateY(-2px); }
.hiw { position:relative; }
.himg { width:100%; height:160px; object-fit:cover; display:block; }
.hbadge { position:absolute; top:.5rem; right:.5rem; }
.hbody { padding:.9rem; }
.hdate { font-size:.75rem; color:var(--ink-m); margin-bottom:.6rem; }
.hnote { font-size:.78rem; color:var(--rose); margin-top:.4rem; }
.hnotes { font-size:.78rem; color:var(--ink-m); font-style:italic;
  margin-top:.4rem; border-top:1px solid var(--border); padding-top:.4rem; }
.hchg { font-size:.78rem; font-weight:500; margin-top:.4rem; }
.hchg.pos { color:var(--sage); } .hchg.neg { color:var(--rose); }
.del-form { margin-top:.75rem; }
.mini-scores { display:grid; grid-template-columns:1fr 1fr; gap:.35rem; }
.mini-score  { background:var(--parch); border-radius:5px; padding:.3rem .5rem;
  display:flex; justify-content:space-between; align-items:center; }
.ms-label { font-size:.68rem; color:var(--ink-m); text-transform:uppercase; letter-spacing:.05em; }
.ms-val   { font-family:'Cormorant Garamond',serif; font-size:1rem;
  font-weight:500; color:var(--sage-d); }

/* ── Data rights ── */
.data-rights h3 { font-size:1.1rem; margin-bottom:.6rem; }
.data-rights p  { font-size:.875rem; color:var(--ink-l); line-height:1.6; margin-bottom:1rem; }

/* ── Footer ── */
.site-footer { border-top:1px solid var(--border); background:var(--ww);
  padding:1.25rem 2rem; text-align:center; }
.disclaimer  { font-size:.78rem; color:var(--rose); background:#fdf2f0;
  border:1px solid #e8c0b8; border-radius:var(--rs); padding:.6rem 1rem;
  max-width:700px; margin:0 auto .75rem; line-height:1.5; }
.footer-copy { font-size:.75rem; color:var(--ink-m); }
"""

# ── Shared JS ────────────────────────────────────────────────────────────────
_JS = """
document.addEventListener('DOMContentLoaded', () => {
  // Auto-dismiss flash messages
  document.querySelectorAll('.flash').forEach(el => {
    setTimeout(() => {
      el.style.transition = 'opacity .4s';
      el.style.opacity    = '0';
      setTimeout(() => el.remove(), 400);
    }, 5000);
  });
  // Animate bar fills
  document.querySelectorAll('.bfill').forEach(b => {
    const t = b.style.width;
    b.style.width = '0';
    requestAnimationFrame(() => setTimeout(() => { b.style.width = t; }, 100));
  });
  // Animate SVG ring
  document.querySelectorAll('.rsv circle:last-child').forEach(c => {
    const t = parseFloat(c.getAttribute('stroke-dashoffset') || '0');
    c.setAttribute('stroke-dashoffset', '314');
    c.style.transition = 'stroke-dashoffset 1s ease';
    setTimeout(() => c.setAttribute('stroke-dashoffset', t), 200);
  });
});
"""

# ── Base wrapper ─────────────────────────────────────────────────────────────
def _base(title, active, head_extra, content, scripts_extra=''):
    nav = ''
    if True:  # always render nav structure (Jinja will hide if not auth)
        nav = f"""
<nav class="navbar">
  <a class="nav-brand" href="/dashboard"><span class="brand-mark">◈</span> SanjuAI</a>
  <div class="nav-links">
    <a href="/dashboard" {'class="active"' if active=='dash' else ''}>Dashboard</a>
    <a href="/capture"   {'class="active"' if active=='cap'  else ''}>New Scan</a>
    <a href="/history"   {'class="active"' if active=='hist' else ''}>History</a>
  </div>
  <div class="nav-user">
    <span class="user-chip">{{% if current_user.is_authenticated %}}{{{{ current_user.username }}}}{{% endif %}}</span>
    <a href="/logout" class="btn-ghost-sm">Log out</a>
  </div>
</nav>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>{_CSS}</style>
  {head_extra}
</head>
<body>
{{% if current_user.is_authenticated %}}
{nav}
{{% endif %}}
<div class="flash-container">
  {{% with messages = get_flashed_messages(with_categories=true) %}}
  {{% for cat, msg in messages %}}
  <div class="flash flash-{{{{ cat }}}}">{{{{ msg }}}}
    <button class="flash-close" onclick="this.parentElement.remove()">×</button>
  </div>
  {{% endfor %}}{{% endwith %}}
</div>
<main>{content}</main>
<footer class="site-footer">
  <div class="disclaimer">
    <strong>⚠ Medical Disclaimer:</strong> SanjuAI is a skincare product
    effectiveness tracking tool only. It does <em>not</em> diagnose, treat,
    or provide medical advice. Consult a qualified Sanjuatologist for any skin concerns.
  </div>
  <p class="footer-copy">© 2026 SanjuAI · COM668 Computing Project · B00912171</p>
</footer>
<script>{_JS}</script>
{scripts_extra}
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════
# SECTION 9 — HTML TEMPLATES
# ═══════════════════════════════════════════════════════════════════

INDEX_T = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SanjuAI — Skincare Progress Tracker</title>
  <style>""" + _CSS + """</style>
</head>
<body>
<div class="flash-container">
  {% with messages = get_flashed_messages(with_categories=true) %}
  {% for cat, msg in messages %}
  <div class="flash flash-{{ cat }}">{{ msg }}
    <button class="flash-close" onclick="this.parentElement.remove()">×</button>
  </div>
  {% endfor %}{% endwith %}
</div>

<div class="auth-layout">
  <div class="auth-hero">
    <div class="hero-inner">
      <p class="hero-eyebrow">COM668 Computing Project</p>
      <h1 class="hero-title">Sanju<em>AI</em></h1>
      <p class="hero-sub">Objective, AI-powered skincare progress tracking.<br>Measure what matters. See real results.</p>
      <div class="hero-feats">
        <div class="feat"><span class="feat-icon">◎</span><span>Face detection &amp; alignment</span></div>
        <div class="feat"><span class="feat-icon">◎</span><span>Acne, redness &amp; texture analysis</span></div>
        <div class="feat"><span class="feat-icon">◎</span><span>Progress charts over time</span></div>
        <div class="feat"><span class="feat-icon">◎</span><span>All data stays on your device</span></div>
      </div>
    </div>
  </div>

  <div class="auth-forms">
    <!-- LOGIN -->
    <div class="form-card" id="loginCard" {% if show_register %}style="display:none"{% endif %}>
      <h2 class="form-title">Welcome back</h2>
      <p class="form-sub">Sign in to your account</p>
      <form method="POST" action="/login">
        <div class="field"><label>Username</label>
          <input type="text" name="username" placeholder="your username" required autofocus></div>
        <div class="field"><label>Password</label>
          <input type="password" name="password" placeholder="••••••••" required></div>
        <button type="submit" class="btn-primary fw">Sign In</button>
      </form>
      <p class="form-switch">New here? <a href="#" onclick="toggleForms()">Create an account</a></p>
    </div>

    <!-- REGISTER -->
    <div class="form-card" id="registerCard" {% if not show_register %}style="display:none"{% endif %}>
      <h2 class="form-title">Create account</h2>
      <p class="form-sub">Start tracking your skincare journey</p>
      <form method="POST" action="/register">
        <div class="field"><label>Username</label>
          <input type="text" name="username" placeholder="choose a username" required></div>
        <div class="field"><label>Email</label>
          <input type="email" name="email" placeholder="you@example.com" required></div>
        <div class="field"><label>Password</label>
          <input type="password" name="password" placeholder="at least 6 characters" required minlength="6"></div>
        <div class="field"><label>Confirm Password</label>
          <input type="password" name="confirm_password" placeholder="repeat password" required></div>
        <div class="consent-box">
          <label class="consent-label">
            <input type="checkbox" name="consent" required>
            <span>I understand this application captures and locally stores facial images for skincare
            analysis. I consent to this use and acknowledge SanjuAI is <strong>not a medical device</strong>
            and provides no medical advice.</span>
          </label>
        </div>
        <button type="submit" class="btn-primary fw">Create Account</button>
      </form>
      <p class="form-switch">Already have an account? <a href="#" onclick="toggleForms()">Sign in</a></p>
    </div>

    <p class="disc-mini">⚠ This tool tracks skincare product progress only. It does not diagnose
    or treat skin conditions. Consult a Sanjuatologist for medical concerns.</p>
  </div>
</div>

<script>
function toggleForms() {
  const l = document.getElementById('loginCard'),
        r = document.getElementById('registerCard');
  if (l.style.display === 'none') { l.style.display='block'; r.style.display='none'; }
  else { l.style.display='none'; r.style.display='block'; }
}
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.flash').forEach(el => {
    setTimeout(() => { el.style.transition='opacity .4s'; el.style.opacity='0';
      setTimeout(() => el.remove(), 400); }, 5000);
  });
});
</script>
</body>
</html>"""


DASHBOARD_T = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Dashboard — SanjuAI</title>
  <style>""" + _CSS + """</style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
</head>
<body>
{% if current_user.is_authenticated %}
<nav class="navbar">
  <a class="nav-brand" href="/dashboard"><span class="brand-mark">◈</span> SanjuAI</a>
  <div class="nav-links">
    <a href="/dashboard" class="active">Dashboard</a>
    <a href="/capture">New Scan</a>
    <a href="/history">History</a>
  </div>
  <div class="nav-user">
    <span class="user-chip">{{ current_user.username }}</span>
    <a href="/logout" class="btn-ghost-sm">Log out</a>
  </div>
</nav>
{% endif %}
<div class="flash-container">
  {% with messages = get_flashed_messages(with_categories=true) %}
  {% for cat, msg in messages %}
  <div class="flash flash-{{ cat }}">{{ msg }}
    <button class="flash-close" onclick="this.parentElement.remove()">×</button>
  </div>
  {% endfor %}{% endwith %}
</div>
<main>
<div class="pc">
  <div class="ph">
    <div>
      <h1 class="pt">Skin Health Dashboard</h1>
      <p class="ps">{{ total_scans }} scan{{ 's' if total_scans != 1 }} recorded</p>
    </div>
    <a href="/capture" class="btn-primary">+ New Scan</a>
  </div>

  {% if not latest %}
  <div class="empty-state">
    <div class="empty-icon">◈</div>
    <h2>No scans yet</h2>
    <p>Capture your first facial scan to start tracking your skincare progress.</p>
    <a href="/capture" class="btn-primary">Capture First Scan</a>
  </div>

  {% else %}

  <!-- Score Cards -->
  <div class="sg">
    <div class="sc sc--o">
      <p class="sl">Overall Score</p>
      <div class="sring">
        <svg viewBox="0 0 120 120" class="rsv">
          <circle cx="60" cy="60" r="50" fill="none" stroke="rgba(255,255,255,.15)" stroke-width="10"/>
          <circle cx="60" cy="60" r="50" fill="none" stroke="rgba(255,255,255,.9)" stroke-width="10"
                  stroke-linecap="round" stroke-dasharray="314"
                  stroke-dashoffset="{{ 314 - (latest.overall_score / 100 * 314) }}"
                  transform="rotate(-90 60 60)"/>
        </svg>
        <span class="rv">{{ latest.overall_score | int }}</span>
      </div>
      <div class="schg {% if latest.overall_change >= 0 %}pos{% else %}neg{% endif %}">
        {% if latest.overall_change > 0 %}▲{% elif latest.overall_change < 0 %}▼{% else %}—{% endif %}
        {{ latest.overall_change | abs | round(1) }}
      </div>
    </div>

    {% for label, score, change, color in [
      ('Skin Clarity', latest.acne_score,    latest.acne_change,    'var(--sage)'),
      ('Redness',      latest.redness_score, latest.redness_change, 'var(--rose)'),
      ('Texture',      latest.texture_score, latest.texture_change, 'var(--sky)'),
    ] %}
    <div class="sc">
      <p class="sl">{{ label }}</p>
      <div class="bscore">
        <div class="btrack"><div class="bfill" style="width:{{ score }}%; background:{{ color }}"></div></div>
        <span class="bnum">{{ score | int }}</span>
      </div>
      <div class="schg {% if change >= 0 %}pos{% else %}neg{% endif %}">
        {% if change > 0 %}▲{% elif change < 0 %}▼{% else %}—{% endif %}
        {{ change | abs | round(1) }} vs baseline
      </div>
    </div>
    {% endfor %}
  </div>

  <!-- Latest & Compare -->
  <div class="sr2">
    <div class="card">
      <h3 class="ct">Latest Scan</h3>
      <div class="sprev">
        <img src="/uploads/{{ latest.image_filename }}" class="simg" alt="Latest scan">
        <div class="smeta">
          <p><strong>Captured:</strong> {{ latest.captured_at.strftime('%d %b %Y, %H:%M') }}</p>
          <p><strong>Acne spots:</strong> {{ latest.acne_count | int }}</p>
          <p><strong>Status:</strong>
            <span class="badge badge--{{ latest.analysis_status }}">{{ latest.analysis_status }}</span>
          </p>
          {% if latest.notes %}<p><strong>Notes:</strong> {{ latest.notes }}</p>{% endif %}
        </div>
      </div>
    </div>

    {% if total_scans > 1 %}
    <div class="card">
      <h3 class="ct">Progress vs Baseline</h3>
      <div class="compare-row">
        <div class="compare-col">
          <p class="compare-label">Baseline</p>
          <img src="/uploads/{{ baseline.image_filename }}" class="compare-img" alt="Baseline">
          <p class="compare-date">{{ baseline.captured_at.strftime('%d %b %Y') }}</p>
        </div>
        <div class="compare-arrow">→</div>
        <div class="compare-col">
          <p class="compare-label">Latest</p>
          <img src="/uploads/{{ latest.image_filename }}" class="compare-img" alt="Latest">
          <p class="compare-date">{{ latest.captured_at.strftime('%d %b %Y') }}</p>
        </div>
      </div>
    </div>
    {% endif %}
  </div>

  <!-- Charts -->
  {% if total_scans > 1 %}
  <div class="card">
    <h3 class="ct">Progress Over Time</h3>
    <div class="chtabs">
      <button class="tb active" onclick="showChart('cO', this)">Overall</button>
      <button class="tb"        onclick="showChart('cA', this)">Skin Clarity</button>
      <button class="tb"        onclick="showChart('cR', this)">Redness</button>
      <button class="tb"        onclick="showChart('cT', this)">Texture</button>
    </div>
    <div class="chtwrap">
      <canvas id="cO"></canvas>
      <canvas id="cA" style="display:none"></canvas>
      <canvas id="cR" style="display:none"></canvas>
      <canvas id="cT" style="display:none"></canvas>
    </div>
  </div>
  {% endif %}

  {% endif %}
</div>
</main>
<footer class="site-footer">
  <div class="disclaimer"><strong>⚠ Medical Disclaimer:</strong> SanjuAI tracks skincare
  effectiveness only — not a medical device. Consult a Sanjuatologist for skin health concerns.</div>
  <p class="footer-copy">© 2026 SanjuAI · COM668 Computing Project · B00912171</p>
</footer>
<script>""" + _JS + """</script>
{% if total_scans > 1 %}
<script>
const L = {{ chart_labels | tojson }};
const cfg = { type:'line', options:{ responsive:true,
  plugins:{ legend:{ display:false } },
  scales:{ x:{ grid:{ color:'rgba(0,0,0,.05)' } },
           y:{ min:0, max:100, grid:{ color:'rgba(0,0,0,.05)' } } },
  elements:{ point:{ radius:5, hoverRadius:7 } }
}};
function mk(id, data, c) {
  return new Chart(document.getElementById(id), { ...cfg,
    data:{ labels:L, datasets:[{ data, borderColor:c,
      backgroundColor: c+'22', fill:true, tension:.4, borderWidth:2.5 }] }
  });
}
mk('cO', {{ overall_data  | tojson }}, '#4a7c59');
mk('cA', {{ acne_data     | tojson }}, '#4a7c59');
mk('cR', {{ redness_data  | tojson }}, '#c17b6c');
mk('cT', {{ texture_data  | tojson }}, '#6c9ec1');

function showChart(id, btn) {
  ['cO','cA','cR','cT'].forEach(k =>
    document.getElementById(k).style.display = k === id ? 'block' : 'none');
  document.querySelectorAll('.tb').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}
</script>
{% endif %}
</body>
</html>"""


CAPTURE_T = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>New Scan — SanjuAI</title>
  <style>""" + _CSS + """</style>
</head>
<body>
{% if current_user.is_authenticated %}
<nav class="navbar">
  <a class="nav-brand" href="/dashboard"><span class="brand-mark">◈</span> SanjuAI</a>
  <div class="nav-links">
    <a href="/dashboard">Dashboard</a>
    <a href="/capture" class="active">New Scan</a>
    <a href="/history">History</a>
  </div>
  <div class="nav-user">
    <span class="user-chip">{{ current_user.username }}</span>
    <a href="/logout" class="btn-ghost-sm">Log out</a>
  </div>
</nav>
{% endif %}
<div class="flash-container">
  {% with messages = get_flashed_messages(with_categories=true) %}
  {% for cat, msg in messages %}
  <div class="flash flash-{{ cat }}">{{ msg }}
    <button class="flash-close" onclick="this.parentElement.remove()">×</button>
  </div>
  {% endfor %}{% endwith %}
</div>
<main>
<div class="pc pc--n">
  <div class="ph">
    <div>
      <h1 class="pt">New Skin Scan</h1>
      <p class="ps">Upload a clear, frontal photo of your face for analysis</p>
    </div>
  </div>

  <div class="tips-bar">
    <div class="tip"><span></span> Face the camera directly</div>
    <div class="tip"><span></span> Good natural or bright light</div>
    <div class="tip"><span></span> Keep consistent distance</div>
    <div class="tip"><span></span> Clean, make-up-free skin</div>
  </div>

  <div class="card">
    <form method="POST" action="/upload" enctype="multipart/form-data" id="uploadForm">
      <div class="upload-zone" id="uploadZone">
        <input type="file" name="image" id="imageInput" accept="image/*"
               style="display:none" onchange="previewImage(event)" required>
        <div id="uploadPlaceholder" onclick="document.getElementById('imageInput').click()">
          <div class="upload-icon">◈</div>
          <p class="upload-cta">Click to choose an image</p>
          <p class="upload-hint">JPG, PNG or WEBP · max 16 MB</p>
        </div>
        <div id="previewContainer" style="display:none">
          <img id="previewImg" src="" alt="Preview" class="preview-image">
          <div class="preview-actions">
            <button type="button" class="btn-ghost" onclick="clearPreview()">Choose different</button>
          </div>
        </div>
      </div>

      <div class="field">
        <label for="notes">Notes <span class="opt">(optional)</span></label>
        <input type="text" name="notes" id="notes"
               placeholder="e.g. Week 3 of new moisturiser, morning routine…">
      </div>

      <div class="consent-reminder">
        By submitting this scan you confirm consent to facial image capture
        for personal skincare tracking. All data is stored locally only.
      </div>

      <button type="submit" class="btn-primary fw" id="submitBtn" disabled>
        <span id="btnText">Analyse Scan</span>
        <span id="btnLoader" style="display:none">Analysing… please wait</span>
      </button>
    </form>
  </div>

  <div class="info-grid">
    <div class="info-card">
      <h4>◎ Acne Detection</h4>
      <p>Counts blemishes and measures skin clarity using blob detection
         and HSV colour analysis.</p>
    </div>
    <div class="info-card">
      <h4>◎ Redness Analysis</h4>
      <p>Measures inflammation by analysing colour distribution
         across all facial regions.</p>
    </div>
    <div class="info-card">
      <h4>◎ Texture Score</h4>
      <p>Evaluates skin smoothness using Local Binary Pattern (LBP)
         texture descriptors (Ojala et al., 2002).</p>
    </div>
  </div>
</div>
</main>
<footer class="site-footer">
  <div class="disclaimer"><strong>⚠ Medical Disclaimer:</strong> SanjuAI tracks skincare
  effectiveness only — not a medical device.</div>
  <p class="footer-copy">© 2026 SanjuAI · COM668 Computing Project · B00912171</p>
</footer>
<script>""" + _JS + """</script>
<script>
function previewImage(e) {
  const f = e.target.files[0]; if (!f) return;
  const r = new FileReader();
  r.onload = ev => {
    document.getElementById('previewImg').src = ev.target.result;
    document.getElementById('uploadPlaceholder').style.display = 'none';
    document.getElementById('previewContainer').style.display  = 'block';
    document.getElementById('submitBtn').disabled = false;
  };
  r.readAsDataURL(f);
}
function clearPreview() {
  document.getElementById('imageInput').value = '';
  document.getElementById('previewImg').src   = '';
  document.getElementById('uploadPlaceholder').style.display = 'block';
  document.getElementById('previewContainer').style.display  = 'none';
  document.getElementById('submitBtn').disabled = true;
}
document.getElementById('uploadForm').addEventListener('submit', () => {
  document.getElementById('btnText').style.display   = 'none';
  document.getElementById('btnLoader').style.display = 'inline';
  document.getElementById('submitBtn').disabled      = true;
});
const z = document.getElementById('uploadZone');
z.addEventListener('dragover', e => { e.preventDefault(); z.classList.add('drag-over'); });
z.addEventListener('dragleave', () => z.classList.remove('drag-over'));
z.addEventListener('drop', e => {
  e.preventDefault(); z.classList.remove('drag-over');
  const inp = document.getElementById('imageInput');
  const dt  = new DataTransfer();
  dt.items.add(e.dataTransfer.files[0]);
  inp.files = dt.files;
  previewImage({ target: inp });
});
</script>
</body>
</html>"""


HISTORY_T = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>History — SanjuAI</title>
  <style>""" + _CSS + """</style>
</head>
<body>
{% if current_user.is_authenticated %}
<nav class="navbar">
  <a class="nav-brand" href="/dashboard"><span class="brand-mark">◈</span> SanjuAI</a>
  <div class="nav-links">
    <a href="/dashboard">Dashboard</a>
    <a href="/capture">New Scan</a>
    <a href="/history" class="active">History</a>
  </div>
  <div class="nav-user">
    <span class="user-chip">{{ current_user.username }}</span>
    <a href="/logout" class="btn-ghost-sm">Log out</a>
  </div>
</nav>
{% endif %}
<div class="flash-container">
  {% with messages = get_flashed_messages(with_categories=true) %}
  {% for cat, msg in messages %}
  <div class="flash flash-{{ cat }}">{{ msg }}
    <button class="flash-close" onclick="this.parentElement.remove()">×</button>
  </div>
  {% endfor %}{% endwith %}
</div>
<main>
<div class="pc">
  <div class="ph">
    <div>
      <h1 class="pt">Scan History</h1>
      <p class="ps">{{ scans | length }} total scan{{ 's' if scans | length != 1 }}</p>
    </div>
    <a href="/capture" class="btn-primary">+ New Scan</a>
  </div>

  {% if not scans %}
  <div class="empty-state">
    <div class="empty-icon">◈</div>
    <h2>No scans yet</h2>
    <p>Start your skincare tracking journey by capturing your first scan.</p>
    <a href="/capture" class="btn-primary">Capture First Scan</a>
  </div>

  {% else %}
  <div class="history-grid">
    {% for scan in scans %}
    <div class="hcard">
      <div class="hiw">
        <img src="/uploads/{{ scan.image_filename }}" class="himg" alt="Scan {{ scan.id }}">
        <span class="badge badge--{{ scan.analysis_status }} hbadge">{{ scan.analysis_status }}</span>
      </div>
      <div class="hbody">
        <p class="hdate">{{ scan.captured_at.strftime('%d %b %Y · %H:%M') }}</p>

        {% if scan.analysis_status == 'complete' %}
        <div class="mini-scores">
          <div class="mini-score"><span class="ms-label">Overall</span>
            <span class="ms-val">{{ scan.overall_score | int }}</span></div>
          <div class="mini-score"><span class="ms-label">Clarity</span>
            <span class="ms-val">{{ scan.acne_score | int }}</span></div>
          <div class="mini-score"><span class="ms-label">Redness</span>
            <span class="ms-val">{{ scan.redness_score | int }}</span></div>
          <div class="mini-score"><span class="ms-label">Texture</span>
            <span class="ms-val">{{ scan.texture_score | int }}</span></div>
        </div>
        {% if scan.overall_change != 0 %}
        <p class="hchg {% if scan.overall_change > 0 %}pos{% else %}neg{% endif %}">
          {% if scan.overall_change > 0 %}▲{% else %}▼{% endif %}
          {{ scan.overall_change | abs | round(1) }} vs baseline
        </p>
        {% endif %}

        {% elif scan.analysis_status == 'failed' %}
        <p class="hnote">⚠ No face detected — re-capture in better light.</p>
        {% else %}
        <p class="hnote">Pending analysis</p>
        {% endif %}

        {% if scan.notes %}<p class="hnotes">{{ scan.notes }}</p>{% endif %}

        <form method="POST" action="/scan/{{ scan.id }}/delete" class="del-form"
              onsubmit="return confirm('Delete this scan permanently?')">
          <button type="submit" class="btn-dsm">Delete</button>
        </form>
      </div>
    </div>
    {% endfor %}
  </div>

  <div class="card data-rights">
    <h3>Your Data Rights</h3>
    <p>All facial images and analysis data are stored locally on this device only.
       You may delete individual scans above, or permanently delete your entire
       account and all associated data below (GDPR right to erasure).</p>
    <form method="POST" action="/account/delete"
          onsubmit="return confirm('This will permanently delete your account and ALL data. Are you absolutely sure?')">
      <button type="submit" class="btn-danger">Delete My Account &amp; All Data</button>
    </form>
  </div>
  {% endif %}
</div>
</main>
<footer class="site-footer">
  <div class="disclaimer"><strong>⚠ Medical Disclaimer:</strong> SanjuAI tracks skincare
  effectiveness only — not a medical device. Consult a Sanjuatologist for skin concerns.</div>
  <p class="footer-copy">© 2026 SanjuAI · COM668 Computing Project · B00912171</p>
</footer>
<script>""" + _JS + """</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════
# SECTION 10 — ROUTES
# ═══════════════════════════════════════════════════════════════════

def _allowed(filename):
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'webp'})


@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template_string(INDEX_T, show_register=False)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        u  = request.form.get('username', '').strip()
        e  = request.form.get('email', '').strip().lower()
        p  = request.form.get('password', '')
        cp = request.form.get('confirm_password', '')
        c  = request.form.get('consent') == 'on'

        if not c:
            flash('You must provide informed consent to use this application.', 'error')
            return render_template_string(INDEX_T, show_register=True)
        if p != cp:
            flash('Passwords do not match.', 'error')
            return render_template_string(INDEX_T, show_register=True)
        if len(p) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return render_template_string(INDEX_T, show_register=True)
        if User.query.filter_by(username=u).first():
            flash('Username already taken.', 'error')
            return render_template_string(INDEX_T, show_register=True)
        if User.query.filter_by(email=e).first():
            flash('Email already registered.', 'error')
            return render_template_string(INDEX_T, show_register=True)

        user = User(username=u, email=e, consent_given=c)
        user.set_password(p)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        flash(f'Welcome, {u}! Your account has been created.', 'success')
        return redirect(url_for('dashboard'))

    return render_template_string(INDEX_T, show_register=True)


@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        u    = request.form.get('username', '').strip()
        p    = request.form.get('password', '')
        user = User.query.filter_by(username=u).first()
        if user and user.check_password(p):
            login_user(user, remember=True)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password.', 'error')
    return render_template_string(INDEX_T, show_register=False)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


@app.route('/dashboard')
@login_required
def dashboard():
    scans    = (SkinScan.query.filter_by(user_id=current_user.id)
                .order_by(SkinScan.captured_at.asc()).all())
    latest   = scans[-1] if scans else None
    baseline = scans[0]  if scans else None
    return render_template_string(
        DASHBOARD_T,
        scans         = scans,
        latest        = latest,
        baseline      = baseline,
        chart_labels  = [s.captured_at.strftime('%d %b') for s in scans],
        overall_data  = [s.overall_score  for s in scans],
        acne_data     = [s.acne_score     for s in scans],
        redness_data  = [s.redness_score  for s in scans],
        texture_data  = [s.texture_score  for s in scans],
        total_scans   = len(scans),
    )


@app.route('/capture')
@login_required
def capture_page():
    return render_template_string(CAPTURE_T)


@app.route('/upload', methods=['POST'])
@login_required
def upload():
    if 'image' not in request.files:
        flash('No image provided.', 'error')
        return redirect(url_for('capture_page'))

    file  = request.files['image']
    notes = request.form.get('notes', '')

    if not file.filename or not _allowed(file.filename):
        flash('Unsupported file type. Use PNG, JPG or JPEG.', 'error')
        return redirect(url_for('capture_page'))

    ext      = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{current_user.id}_{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    scan = SkinScan(user_id=current_user.id, image_filename=filename,
                    notes=notes, analysis_status='pending')
    db.session.add(scan)
    db.session.commit()

    bl = (SkinScan.query
          .filter_by(user_id=current_user.id, analysis_status='complete')
          .order_by(SkinScan.captured_at.asc()).first())
    if bl and bl.id == scan.id:
        bl = None

    try:
        res = analyse_image(filepath, baseline_scan=bl)
        scan.face_detected   = res['face_detected']
        scan.acne_count      = res['acne_count']
        scan.acne_score      = res['acne_score']
        scan.redness_score   = res['redness_score']
        scan.texture_score   = res['texture_score']
        scan.overall_score   = res['overall_score']
        scan.acne_change     = res['acne_change']
        scan.redness_change  = res['redness_change']
        scan.texture_change  = res['texture_change']
        scan.overall_change  = res['overall_change']
        scan.analysis_status = res['analysis_status']
        db.session.commit()

        if not res['face_detected']:
            flash(f'⚠ {res["message"]} — scan saved but not analysed.', 'warning')
        else:
            flash('✓ Scan analysed successfully!', 'success')
    except Exception as e:
        scan.analysis_status = 'failed'
        db.session.commit()
        flash(f'Analysis error: {e}', 'error')

    return redirect(url_for('dashboard'))


@app.route('/history')
@login_required
def history():
    scans = (SkinScan.query.filter_by(user_id=current_user.id)
             .order_by(SkinScan.captured_at.desc()).all())
    return render_template_string(HISTORY_T, scans=scans)


@app.route('/scan/<int:scan_id>/delete', methods=['POST'])
@login_required
def delete_scan(scan_id):
    scan = SkinScan.query.get_or_404(scan_id)
    if scan.user_id != current_user.id:
        abort(403)
    fp = os.path.join(UPLOAD_DIR, scan.image_filename)
    if os.path.exists(fp):
        os.remove(fp)
    db.session.delete(scan)
    db.session.commit()
    flash('Scan deleted.', 'info')
    return redirect(url_for('history'))


@app.route('/account/delete', methods=['POST'])
@login_required
def delete_account():
    """GDPR right to erasure — deletes all user data."""
    user = current_user
    for scan in user.scans:
        fp = os.path.join(UPLOAD_DIR, scan.image_filename)
        if os.path.exists(fp):
            os.remove(fp)
    logout_user()
    db.session.delete(user)
    db.session.commit()
    flash('Your account and all data have been permanently deleted.', 'info')
    return redirect(url_for('index'))


@app.route('/api/scans')
@login_required
def api_scans():
    scans = (SkinScan.query.filter_by(user_id=current_user.id)
             .order_by(SkinScan.captured_at.asc()).all())
    return jsonify([s.to_dict() for s in scans])


# ═══════════════════════════════════════════════════════════════════
# SECTION 11 — ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        logger.info("Database tables created / verified.")

    print("=" * 60)
    print("  SanjuAI — AI Skincare Progress Tracker")
    print("  COM668 Computing Project · B00912171")
    print("  ----------------------------------------")
    print("  Open your browser:  http://127.0.0.1:5000")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)
