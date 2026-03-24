import os
import uuid
from datetime import datetime

from flask import (Blueprint, render_template, request, redirect,
                   url_for, flash, jsonify, current_app, abort)
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename

from . import db
from .models import User, SkinScan
from .processing.scorer import analyse_image

main = Blueprint('main', __name__)


def _allowed_file(filename):
    exts = current_app.config.get('ALLOWED_EXTENSIONS', {'png','jpg','jpeg','webp'})
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in exts


# ─────────────────────────────────────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────────────────────────────────────

@main.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    return render_template('index.html')


@main.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))

    if request.method == 'POST':
        username  = request.form.get('username', '').strip()
        email     = request.form.get('email', '').strip().lower()
        password  = request.form.get('password', '')
        confirm   = request.form.get('confirm_password', '')
        consent   = request.form.get('consent') == 'on'

        if not consent:
            flash('You must provide informed consent to use this application.', 'error')
            return render_template('index.html', show_register=True)

        if password != confirm:
            flash('Passwords do not match.', 'error')
            return render_template('index.html', show_register=True)

        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return render_template('index.html', show_register=True)

        if User.query.filter_by(username=username).first():
            flash('Username already taken.', 'error')
            return render_template('index.html', show_register=True)

        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'error')
            return render_template('index.html', show_register=True)

        user = User(username=username, email=email, consent_given=consent)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        login_user(user)
        flash(f'Welcome, {username}! Your account has been created.', 'success')
        return redirect(url_for('main.dashboard'))

    return render_template('index.html', show_register=True)


@main.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user     = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user, remember=True)
            return redirect(url_for('main.dashboard'))

        flash('Invalid username or password.', 'error')

    return render_template('index.html')


@main.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('main.index'))


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

@main.route('/dashboard')
@login_required
def dashboard():
    scans = (SkinScan.query
             .filter_by(user_id=current_user.id)
             .order_by(SkinScan.captured_at.asc())
             .all())

    latest  = scans[-1] if scans else None
    baseline = scans[0]  if scans else None

    # Build chart data
    chart_labels  = [s.captured_at.strftime('%d %b') for s in scans]
    overall_data  = [s.overall_score  for s in scans]
    acne_data     = [s.acne_score     for s in scans]
    redness_data  = [s.redness_score  for s in scans]
    texture_data  = [s.texture_score  for s in scans]

    return render_template(
        'dashboard.html',
        scans         = scans,
        latest        = latest,
        baseline      = baseline,
        chart_labels  = chart_labels,
        overall_data  = overall_data,
        acne_data     = acne_data,
        redness_data  = redness_data,
        texture_data  = texture_data,
        total_scans   = len(scans),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CAPTURE / UPLOAD
# ─────────────────────────────────────────────────────────────────────────────

@main.route('/capture')
@login_required
def capture():
    return render_template('capture.html')


@main.route('/upload', methods=['POST'])
@login_required
def upload():
    if 'image' not in request.files:
        flash('No image file provided.', 'error')
        return redirect(url_for('main.capture'))

    file  = request.files['image']
    notes = request.form.get('notes', '')

    if file.filename == '':
        flash('No file selected.', 'error')
        return redirect(url_for('main.capture'))

    if not _allowed_file(file.filename):
        flash('Unsupported file type. Please upload PNG, JPG, or JPEG.', 'error')
        return redirect(url_for('main.capture'))

    # Save with unique filename
    ext      = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{current_user.id}_{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Create DB record
    scan = SkinScan(
        user_id        = current_user.id,
        image_filename = filename,
        notes          = notes,
        analysis_status = 'pending',
    )
    db.session.add(scan)
    db.session.commit()

    # Run analysis
    baseline = (SkinScan.query
                .filter_by(user_id=current_user.id, analysis_status='complete')
                .order_by(SkinScan.captured_at.asc())
                .first())

    # Don't use current scan as its own baseline
    if baseline and baseline.id == scan.id:
        baseline = None

    try:
        results = analyse_image(filepath, baseline_scan=baseline)

        scan.face_detected   = results['face_detected']
        scan.acne_count      = results['acne_count']
        scan.acne_score      = results['acne_score']
        scan.redness_score   = results['redness_score']
        scan.texture_score   = results['texture_score']
        scan.overall_score   = results['overall_score']
        scan.acne_change     = results['acne_change']
        scan.redness_change  = results['redness_change']
        scan.texture_change  = results['texture_change']
        scan.overall_change  = results['overall_change']
        scan.analysis_status = results['analysis_status']

        db.session.commit()

        if not results['face_detected']:
            flash(f'⚠ No face detected: {results["message"]} — scan saved but not analysed.', 'warning')
        else:
            flash('✓ Scan captured and analysed successfully!', 'success')

    except Exception as e:
        scan.analysis_status = 'failed'
        db.session.commit()
        flash(f'Analysis error: {e}', 'error')

    return redirect(url_for('main.dashboard'))


# ─────────────────────────────────────────────────────────────────────────────
# HISTORY
# ─────────────────────────────────────────────────────────────────────────────

@main.route('/history')
@login_required
def history():
    scans = (SkinScan.query
             .filter_by(user_id=current_user.id)
             .order_by(SkinScan.captured_at.desc())
             .all())
    return render_template('history.html', scans=scans)


@main.route('/scan/<int:scan_id>/delete', methods=['POST'])
@login_required
def delete_scan(scan_id):
    scan = SkinScan.query.get_or_404(scan_id)
    if scan.user_id != current_user.id:
        abort(403)

    # Delete image file
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], scan.image_filename)
    if os.path.exists(filepath):
        os.remove(filepath)

    db.session.delete(scan)
    db.session.commit()
    flash('Scan deleted.', 'info')
    return redirect(url_for('main.history'))


# ─────────────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────────────

@main.route('/api/scans')
@login_required
def api_scans():
    scans = (SkinScan.query
             .filter_by(user_id=current_user.id)
             .order_by(SkinScan.captured_at.asc())
             .all())
    return jsonify([s.to_dict() for s in scans])


@main.route('/api/scan/<int:scan_id>')
@login_required
def api_scan_detail(scan_id):
    scan = SkinScan.query.get_or_404(scan_id)
    if scan.user_id != current_user.id:
        abort(403)
    return jsonify(scan.to_dict())


@main.route('/account/delete', methods=['POST'])
@login_required
def delete_account():
    """Allow users to delete all their data (GDPR right to erasure)."""
    user = current_user
    # Delete all images
    for scan in user.scans:
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], scan.image_filename)
        if os.path.exists(filepath):
            os.remove(filepath)

    logout_user()
    db.session.delete(user)
    db.session.commit()
    flash('Your account and all data have been permanently deleted.', 'info')
    return redirect(url_for('main.index'))
