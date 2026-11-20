"""Offline ChArUco calibration and depth diagnostics; no robot/Isaac imports.

Coordinates and lengths are in metres. A transform a->b means p_b=R@p_a+t.
Raw SDK data is kept separate from estimated OpenCV models.
"""
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STREAMS = ('rgb', 'ir_left', 'ir_right', 'depth')
FIELDS = ['frame_id', 'batch_id', 'stream', 'relative_path', 'device_timestamp_us',
          'host_timestamp_ns', 'exposure_sdk_raw', 'gain_sdk_raw', 'split',
          'depth_scale_m_per_unit', 'reference_z_m', 'roi_xywh', 'frontoparallel']


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def make_board(config, measured=True):
    key = 'measured' if measured else 'nominal'
    s = config.get(f'{key}_square_length_m')
    m = config.get(f'{key}_marker_length_m')
    if s is None or m is None or not 0 < m < s:
        raise ValueError('Confirm measured square/marker lengths (0 < marker < square).')
    if not config.get('dictionary') or config.get('legacy_pattern') is None:
        raise ValueError('Board dictionary and legacy pattern must be verified.')
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config['dictionary']))
    ids = config.get('marker_ids')
    if ids is None:
        raise ValueError('Board marker ID ordering is required.')
    board = cv2.aruco.CharucoBoard((config['squares_x'], config['squares_y']),
                                 s, m, dictionary, np.array(ids, dtype=np.int32))
    board.setLegacyPattern(config['legacy_pattern'])
    return board


def gray8(image):
    if image is None:
        raise ValueError('Cannot read image.')
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype == np.uint16:
        lo, hi = np.percentile(image, [1, 99])
        image = np.clip((image.astype(float)-lo)*255/max(hi-lo, 1), 0, 255).astype(np.uint8)
    if image.dtype != np.uint8:
        raise ValueError('Only uint8/uint16 images are supported.')
    return image


def detect(image, board):
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.CharucoDetector(board, detectorParams=params)
    corners, ids, _, _ = detector.detectBoard(gray8(image))
    if ids is None or len(ids) < 8:
        raise ValueError('Fewer than 8 ChArUco corners; move closer or improve image quality.')
    ids = ids.ravel()
    if board.checkCharucoCornersCollinear(ids):
        raise ValueError('Detected corners are collinear.')
    return board.getChessboardCorners()[ids].astype(np.float32), corners.reshape(-1, 2), ids


def init_session(path, board_path, square_mm, marker_mm):
    path = Path(path)
    if path.exists():
        raise ValueError('Session already exists; choose a new name. Do not overwrite captures.')
    board = read_json(board_path)
    board.update(measured_square_length_m=square_mm/1000,
                 measured_marker_length_m=marker_mm/1000, status='MEASURED_BY_USER')
    make_board(board)
    path.mkdir(parents=True)
    for name in (*STREAMS, 'results'):
        (path/name).mkdir()
    write_json(path/'board.json', board)
    write_json(path/'manifest.json', {'schema_version': 2, 'status': 'NOT_RUN',
               'device': None, 'profiles': None, 'batches': [], 'opencv_version': cv2.__version__,
               'processing': {'alignment': 'none', 'software_undistortion': False,
                              'depth_filters': 'none added by tool; firmware settings exported'},
               'timestamp_note': 'device microseconds; host monotonic nanoseconds; not exposure-time synchronization'})
    write_json(path/'factory_calibration.json', {'status': 'NOT_READ'})
    with (path/'frames.csv').open('w', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()


def rows(session, stream=None):
    with (Path(session)/'frames.csv').open(newline='') as f:
        data = list(csv.DictReader(f))
    keys = [(r['frame_id'], r['stream']) for r in data]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate frame/stream rows in manifest.')
    return [r for r in data if stream is None or r['stream'] == stream]


def observations(session, stream, board):
    session = Path(session)
    result, rejected, size = [], [], None
    image_splits = {}
    for row in rows(session, stream):
        if row['split'] not in ('fit', 'validation'):
            continue
        image = cv2.imread(str(session/row['relative_path']), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Missing/unreadable capture: {row['relative_path']}")
        current = (image.shape[1], image.shape[0])
        if size is not None and size != current:
            raise ValueError('Mixed image resolutions in one session.')
        size = current
        image_hash = hashlib.sha256(image.tobytes()).hexdigest()
        if image_hash in image_splits and image_splits[image_hash] != row['split']:
            raise ValueError('Identical image appears in fit and validation; use independent captures.')
        image_splits[image_hash] = row['split']
        try:
            obj, img, ids = detect(image, board)
        except ValueError as e:
            rejected.append({'frame_id': row['frame_id'], 'reason': str(e), 'split': row['split']})
            continue
        result.append(dict(row=row, obj=obj, img=img, ids=ids))
    if not result:
        raise ValueError('No usable images. Capture frames first or check board configuration.')
    return result, rejected, size


def as_model(record):
    k = np.asarray(record['camera_matrix'], dtype=float)
    d = np.asarray(record['distortion_coefficients'], dtype=float)
    if k.shape != (3, 3) or not np.all(np.isfinite(k)) or not np.all(np.isfinite(d)):
        raise ValueError('Invalid camera matrix/distortion.')
    if k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('Nonpositive focal length.')
    return k, d


def factory_model(record):
    """Explicitly support forward Brown/none only, never inverse or modified Brown."""
    model = record['distortion_model']
    if model == 'NONE':
        d = np.zeros(5)
    elif model in ('BROWN_CONRADY', 'BROWN_CONRADY_K6'):
        keys = ['k1', 'k2', 'p1', 'p2', 'k3']
        if model == 'BROWN_CONRADY_K6':
            keys += ['k4', 'k5', 'k6']
        d = [record['distortion'][x] for x in keys]
    else:
        raise ValueError(f'Factory model {model} is not supported by this OpenCV comparison.')
    return as_model({'camera_matrix': record['camera_matrix'], 'distortion_coefficients': d})


def reprojection(obj, img, k, d, rvec=None, tvec=None):
    if rvec is None:
        ok, rvec, tvec = cv2.solvePnP(obj, img, k, d, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise ValueError('PnP failed.')
    projected = cv2.projectPoints(obj, rvec, tvec, k, d)[0].reshape(-1, 2)
    err = np.linalg.norm(projected-img, axis=1)
    return float(np.sqrt(np.mean(err**2))), err, rvec, tvec


def evaluate_views(views, k, d):
    records, errors = [], []
    for v in views:
        rms, e, _, _ = reprojection(v['obj'], v['img'], k, d)
        records.append({'frame_id': v['row']['frame_id'], 'rms_px': rms, 'corners': len(e)})
        errors.extend(e.tolist())
    return {'rms_px': float(np.sqrt(np.mean(np.square(errors)))),
            'p95_corner_error_px': float(np.percentile(errors, 95)), 'per_frame': records}


def intrinsic(session, stream, min_fit=15, min_validation=5, max_rms=.5):
    session = Path(session)
    board = make_board(read_json(session/'board.json'))
    obs, rejected, size = observations(session, stream, board)
    fit = [v for v in obs if v['row']['split'] == 'fit']
    val = [v for v in obs if v['row']['split'] == 'validation']
    if len(fit) < min_fit or len(val) < min_validation:
        raise ValueError(f'Need {min_fit} fit/{min_validation} validation views; found {len(fit)}/{len(val)}. Rejected: {rejected}')
    rms, k, d, rv, tv = cv2.calibrateCamera([v['obj'] for v in fit], [v['img'] for v in fit], size,
                                          None, None, flags=0)
    as_model({'camera_matrix': k, 'distortion_coefficients': d})
    normals = np.array([cv2.Rodrigues(r)[0][:, 2] for r in rv])
    normal_span = float(np.degrees(np.arccos(np.clip(normals@normals.T, -1, 1))).max())
    all_pixels = np.concatenate([v['img'] for v in fit])
    coverage = (np.ptp(all_pixels, axis=0)/np.array(size)).tolist()
    validation = evaluate_views(val, k, d)
    # Numerical/coverage gate, not a device accuracy certification.
    passed = validation['rms_px'] <= max_rms and normal_span >= 15 and min(coverage) >= .6 and not rejected
    model = {'status': 'PASS' if passed else 'REVIEW_REQUIRED', 'scope': 'reprojection/coverage gate only',
             'stream': stream, 'image_size_px': size, 'camera_matrix': k.tolist(),
             'distortion_model': 'opencv_brown5', 'coefficient_order': ['k1', 'k2', 'p1', 'p2', 'k3'],
             'distortion_coefficients': d.ravel().tolist(), 'fit_rms_px': float(rms),
             'normal_span_deg': normal_span, 'image_coverage_xy': coverage,
             'fit_frame_ids': [v['row']['frame_id'] for v in fit], 'opencv_version': cv2.__version__,
             'board_sha256': digest(session/'board.json'), 'frames_sha256': digest(session/'frames.csv')}
    manifest = read_json(session/'manifest.json')
    model['device'] = manifest.get('device')
    model['profile'] = (manifest.get('profiles') or {}).get(stream)
    model['processing'] = manifest.get('processing')
    model['device_settings'] = manifest.get('device_settings')
    report = {'status': model['status'], 'threshold_rms_px': max_rms, 'estimated': validation,
              'rejected': rejected, 'validation_frame_ids': [v['row']['frame_id'] for v in val]}
    factory = read_json(session/'factory_calibration.json').get('streams', {}).get(stream)
    if factory:
        try:
            fk, fd = factory_model(factory)
            report['factory'] = evaluate_views(val, fk, fd)
        except ValueError as e:
            report['factory'] = {'status': 'UNSUPPORTED', 'reason': str(e)}
    else:
        report['factory'] = {'status': 'NOT_AVAILABLE'}
    write_json(session/f'results/{stream}_calibration.json', model)
    write_json(session/f'results/{stream}_validation.json', report)
    return model, report


def pair_views(first, second, max_skew_ms):
    other = {v['row']['frame_id']: v for v in second}
    pairs, rejected = [], []
    for a in first:
        frame_id = a['row']['frame_id']
        b = other.get(frame_id)
        reason = None
        if b is None:
            reason = 'other stream not detected'
        elif a['row']['split'] != b['row']['split']:
            reason = 'split mismatch'
        elif abs(int(a['row']['device_timestamp_us'])-int(b['row']['device_timestamp_us'])) > max_skew_ms*1000:
            reason = 'timestamp skew exceeds threshold'
        if reason:
            rejected.append({'frame_id': frame_id, 'reason': reason})
            continue
        common, ia, ib = np.intersect1d(a['ids'], b['ids'], return_indices=True)
        obj = a['obj'][ia]
        if len(common) < 8 or np.linalg.matrix_rank(obj-obj.mean(axis=0), tol=1e-6) < 2:
            rejected.append({'frame_id': frame_id, 'reason': 'insufficient/noncollinear coverage of common corners'})
            continue
        pairs.append({'obj': obj, 'a': a['img'][ia], 'b': b['img'][ib], 'row': a['row']})
    return pairs, rejected


def stereo(session, first, second, min_fit=15, min_validation=5, max_skew_ms=10, max_rms=1):
    session = Path(session)
    if first == second or 'depth' in (first, second):
        raise ValueError('Stereo solve requires two distinct RGB/IR intensity streams, not depth.')
    board = make_board(read_json(session/'board.json'))
    a, ra, size_a = observations(session, first, board)
    b, rb, size_b = observations(session, second, board)
    models = [read_json(session/f'results/{s}_calibration.json') for s in (first, second)]
    for model, size in zip(models, (size_a, size_b)):
        if model['board_sha256'] != digest(session/'board.json') or model['frames_sha256'] != digest(session/'frames.csv') or list(size) != model['image_size_px']:
            raise ValueError('Intrinsic calibration board/size mismatch; recalibrate.')
    ka, da = as_model(models[0]); kb, db = as_model(models[1])
    pairs, rejected = pair_views(a, b, max_skew_ms)
    fit = [p for p in pairs if p['row']['split'] == 'fit']
    val = [p for p in pairs if p['row']['split'] == 'validation']
    if len(fit) < min_fit or len(val) < min_validation:
        raise ValueError(f'Need {min_fit}/{min_validation} paired fit/validation views; found {len(fit)}/{len(val)}.')
    result = cv2.stereoCalibrate([p['obj'] for p in fit], [p['a'] for p in fit], [p['b'] for p in fit],
                                ka, da, kb, db, size_a, flags=cv2.CALIB_FIX_INTRINSIC,
                                criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT, 100, 1e-8))
    rms, _, _, _, _, r, t, _, _ = result
    def evaluate(rotation, translation):
        errors, records = [], []
        for p in val:
            _, _, rv, tv = reprojection(p['obj'], p['a'], ka, da)
            rbv = cv2.Rodrigues(rotation@cv2.Rodrigues(rv)[0])[0]
            tbv = rotation@tv+np.asarray(translation).reshape(3, 1)
            e_rms, e, _, _ = reprojection(p['obj'], p['b'], kb, db, rbv, tbv)
            errors.extend(e.tolist()); records.append({'frame_id': p['row']['frame_id'], 'rms_px': e_rms})
        return {'rms_px': float(np.sqrt(np.mean(np.square(errors)))), 'per_frame': records}
    v = evaluate(r, t)
    factory = read_json(session/'factory_calibration.json').get('extrinsics', {}).get(f'{first}_to_{second}')
    report = {'status': 'PASS' if v['rms_px'] <= max_rms and not (ra or rb or rejected) and all(m['status']=='PASS' for m in models) else 'REVIEW_REQUIRED',
              'scope': 'held-out cross-camera reprojection only; no hand-eye',
              'from': first, 'to': second, 'equation': 'p_to = R @ p_from + t_m',
              'R': r.tolist(), 't_m': t.ravel().tolist(), 'baseline_m': float(np.linalg.norm(t)),
              'fit_rms_px': float(rms), 'validation': v, 'max_skew_ms': max_skew_ms,
              'threshold_rms_px': max_rms, 'rejected': ra+rb+rejected,
              'frames_sha256': digest(session/'frames.csv'), 'board_sha256': digest(session/'board.json')}
    if factory:
        report['factory_validation'] = evaluate(np.array(factory['R']), np.array(factory['t_m']))
    write_json(session/f'results/{first}_to_{second}.json', report)
    return report


def depth_statistics(raw, scale, roi, reference_z):
    if raw.dtype != np.uint16 or raw.ndim != 2:
        raise ValueError('Depth must be original uint16 PNG, not a color visualization.')
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(reference_z) or reference_z <= 0:
        raise ValueError('Positive depth scale and independent reference Z are required.')
    x, y, w, h = roi
    if min(x, y) < 0 or min(w, h) <= 0 or x+w > raw.shape[1] or y+h > raw.shape[0]:
        raise ValueError('Depth ROI outside image.')
    patch = raw[y:y+h, x:x+w]
    valid = (patch > 0) & (patch < 65535)
    if valid.sum() < 30:
        raise ValueError('Fewer than 30 valid depth pixels in ROI.')
    z = patch[valid].astype(float)*scale
    median = float(np.median(z))
    return {'median_z_m': median, 'reference_z_m': reference_z,
            'bias_m': median-reference_z, 'valid_fraction': float(valid.mean()),
            'spatial_std_m': float(np.std(z)), 'mad_m': float(np.median(np.abs(z-median))),
            'p95_absolute_error_m': float(np.percentile(np.abs(z-reference_z), 95))}


def depth_check(session, max_bias_mm=10, min_valid=.9):
    session = Path(session)
    records, rejected = [], []
    for row in rows(session, 'depth'):
        if not row['reference_z_m']:
            continue
        if row['frontoparallel'] != 'True':
            raise ValueError('Reference Z check requires a frontoparallel stationary plane.')
        try:
            raw = cv2.imread(str(session/row['relative_path']), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise ValueError('Depth image missing.')
            stat = depth_statistics(raw, float(row['depth_scale_m_per_unit']),
                                    json.loads(row['roi_xywh']), float(row['reference_z_m']))
        except (ValueError, TypeError) as e:
            rejected.append({'frame_id': row['frame_id'], 'reason': str(e)})
            continue
        records.append({**stat, 'frame_id': row['frame_id'], 'batch_id': row['batch_id'], 'split': row['split']})
    if not records:
        raise ValueError('No depth/reference measurements. Capture a plane with --reference-z-m and --depth-roi.')
    groups = []
    for bid in sorted({r['batch_id'] for r in records}):
        group = [r for r in records if r['batch_id'] == bid]
        if len({(r['reference_z_m'], r['split']) for r in group}) != 1:
            raise ValueError('Reference/split changed within a depth batch.')
        z = [r['median_z_m'] for r in group]
        groups.append({'batch_id': bid, 'count': len(group), 'split': group[0]['split'],
                       'reference_z_m': group[0]['reference_z_m'], 'median_z_m': float(np.median(z)),
                       'temporal_std_of_roi_medians_m': float(np.std(z)) if len(z)>1 else None})
    val = [r for r in records if r['split'] == 'validation']
    passed = bool(val) and not rejected and all(abs(r['bias_m'])*1000 <= max_bias_mm and r['valid_fraction'] >= min_valid for r in val)
    report = {'status': 'PASS' if passed else 'REVIEW_REQUIRED', 'scope': 'frontoparallel plane axial Z checks, not moving-ball accuracy',
              'max_bias_mm': max_bias_mm, 'min_valid_fraction': min_valid, 'frames': records,
              'batches': groups, 'rejected': rejected, 'correction': {'status': 'NOT_FIT'},
              'frames_sha256': digest(session/'frames.csv')}
    fit_groups = [g for g in groups if g['split'] == 'fit']
    val_groups = [g for g in groups if g['split'] == 'validation']
    if len({g['reference_z_m'] for g in fit_groups}) >= 3 and val_groups:
        x = np.array([g['median_z_m'] for g in fit_groups]); y = np.array([g['reference_z_m'] for g in fit_groups])
        if np.ptp(x) > .05:
            scale, offset = np.linalg.lstsq(np.c_[x, np.ones(len(x))], y, rcond=None)[0]
            report['correction'] = {'status': 'CANDIDATE_ONLY', 'equation': 'z_corrected_m = scale*z_raw_m + offset_m',
                'scale': float(scale), 'offset_m': float(offset), 'fit_measured_range_m': [float(x.min()),float(x.max())],
                'validation': [{'batch_id': g['batch_id'], 'raw_bias_m': g['median_z_m']-g['reference_z_m'],
                                'corrected_bias_m': float(scale*g['median_z_m']+offset-g['reference_z_m'])} for g in val_groups],
                'note': 'No firmware changes. Reference origin/tilt errors can masquerade as sensor bias.'}
    write_json(session/'results/depth_validation.json', report)
    return report


def plane_residuals(raw, scale, kd, dd, r_depth_rgb, t_depth_rgb, board_r_rgb, board_t_rgb, board_size, step=4):
    """Depth->RGB consistency against a ChArUco pose, not independent metrology.

    Intersect depth rays with the RGB board plane to select the ROI independently
    of measured depth. Then compare measured 3D points to that plane.
    """
    if raw is None or raw.dtype != np.uint16 or raw.ndim != 2 or scale <= 0:
        raise ValueError('Expected raw uint16 depth and a positive scale.')
    y,x=np.mgrid[0:raw.shape[0]:step,0:raw.shape[1]:step]
    uv=np.c_[x.ravel(),y.ravel()].astype(np.float64)
    xy=cv2.undistortPoints(uv.reshape(-1,1,2),kd,dd).reshape(-1,2)
    rays=np.c_[xy,np.ones(len(xy))]
    r=np.asarray(r_depth_rgb);t=np.asarray(t_depth_rgb).reshape(3)
    rb=np.asarray(board_r_rgb);tb=np.asarray(board_t_rgb).reshape(3)
    direction=rays@r.T
    normal=rb[:,2]
    denom=direction@normal
    z=np.full(len(rays),np.nan)
    safe=np.abs(denom)>1e-8
    z[safe]=np.dot(tb-t,normal)/denom[safe]
    expected=direction*z[:,None]+t
    board=(expected-tb)@rb
    margin=.01
    selected=(z>0)&(board[:,0]>margin)&(board[:,0]<board_size[0]-margin)&(board[:,1]>margin)&(board[:,1]<board_size[1]-margin)
    if selected.sum()<30:
        raise ValueError('Too little board area visible in depth image; use a larger/closer board.')
    values=raw[y.ravel(),x.ravel()]
    valid=selected&(values>0)&(values<65535)
    if valid.sum()<30:
        raise ValueError('Too few valid depth samples on board.')
    measured=direction[valid]*(values[valid]*scale)[:,None]+t
    errors=(measured-tb)@normal
    return {'samples':int(valid.sum()),'valid_fraction':float(valid.sum()/selected.sum()),
            'plane_median_signed_error_m':float(np.median(errors)),
            'plane_p95_absolute_error_m':float(np.percentile(np.abs(errors),95))}



def factory_check(session, stream, min_validation=5, max_rms=.5):
    """Evaluate factory K/distortion without estimating replacement intrinsics."""
    session = Path(session)
    board = make_board(read_json(session/'board.json'))
    obs, rejected, size = observations(session, stream, board)
    views = [v for v in obs if v['row']['split'] == 'validation']
    rejected = [r for r in rejected if r['split'] == 'validation']
    if len(views) < min_validation:
        raise ValueError(f'Need {min_validation} usable validation images; found {len(views)}.')
    record = read_json(session/'factory_calibration.json')['streams'][stream]
    if record['image_size_px'] != list(size):
        raise ValueError('Factory profile size does not match captured images.')
    k, d = factory_model(record)
    result = evaluate_views(views, k, d)
    pixels = np.concatenate([v['img'] for v in views])
    coverage = (np.ptp(pixels, axis=0)/np.array(size)).tolist()
    normals = np.array([cv2.Rodrigues(reprojection(v['obj'],v['img'],k,d)[2])[0][:,2] for v in views])
    span = float(np.degrees(np.arccos(np.clip(normals@normals.T,-1,1))).max())
    result.update(status='PASS' if result['rms_px']<=max_rms and min(coverage)>=.6 and span>=15 and not rejected else 'REVIEW_REQUIRED',
                  scope='factory reprojection and coverage only; no intrinsic fitting',
                  stream=stream, image_coverage_xy=coverage, normal_span_deg=span,
                  threshold_rms_px=max_rms, rejected=rejected,
                  board_sha256=digest(session/'board.json'),frames_sha256=digest(session/'frames.csv'),
                  factory_sha256=digest(session/'factory_calibration.json'))
    write_json(session/f'results/{stream}_factory_validation.json',result)
    return result


def alignment_check(session,max_skew_ms=10,max_plane_error_mm=10,model_source="factory"):
    session=Path(session)
    board_cfg=read_json(session/'board.json');board=make_board(board_cfg)
    obs,rejected,size=observations(session,'rgb',board)
    factory=read_json(session/'factory_calibration.json')
    if model_source == 'factory':
        model = factory_check(session,'rgb')
        k,d = factory_model(factory['streams']['rgb'])
    elif model_source == 'estimated':
        model=read_json(session/'results/rgb_calibration.json')
        if model['board_sha256']!=digest(session/'board.json') or model['frames_sha256']!=digest(session/'frames.csv') or model['image_size_px']!=list(size):
            raise ValueError('RGB intrinsic calibration does not match this board/image size.')
        k,d=as_model(model)
    else:
        raise ValueError('model_source must be factory or estimated.')
    kd,dd=factory_model(factory['streams']['depth'])
    ext=factory['extrinsics']['depth_to_rgb']
    depth={row['frame_id']:row for row in rows(session,'depth')}
    records=[]
    for v in obs:
        if v['row']['split']!='validation':continue
        fid=v['row']['frame_id'];row=depth.get(fid)
        if row is None:
            rejected.append({'frame_id':fid,'reason':'No paired depth'});continue
        if row['split']!='validation' or abs(int(row['device_timestamp_us'])-int(v['row']['device_timestamp_us']))>max_skew_ms*1000:
            rejected.append({'frame_id':fid,'reason':'Depth/RGB split or timestamp mismatch'});continue
        _,_,rv,tv=reprojection(v['obj'],v['img'],k,d)
        raw=cv2.imread(str(session/row['relative_path']),cv2.IMREAD_UNCHANGED)
        if raw is not None and [raw.shape[1],raw.shape[0]]!=factory['streams']['depth']['image_size_px']:
            raise ValueError('Depth image size differs from factory model.')
        try:
            stats=plane_residuals(raw,float(row['depth_scale_m_per_unit']),kd,dd,ext['R'],ext['t_m'],
                                  cv2.Rodrigues(rv)[0],tv,(board_cfg['squares_x']*board_cfg['measured_square_length_m'],
                                  board_cfg['squares_y']*board_cfg['measured_square_length_m']))
        except ValueError as e:
            rejected.append({'frame_id':fid,'reason':str(e)});continue
        records.append({'frame_id':fid,**stats})
    if not records:raise ValueError('No usable held-out RGB/depth board pairs.')
    passed=len(records)>=5 and not rejected and model['status']=='PASS' and all(r['plane_p95_absolute_error_m']*1000<=max_plane_error_mm and r['valid_fraction']>=.9 for r in records)
    report={'status':'PASS' if passed else 'REVIEW_REQUIRED','scope':'factory depth->RGB plane consistency; NOT an independent metric accuracy or full extrinsic observability test',
            'rgb_model_source':model_source,'factory_sha256':digest(session/'factory_calibration.json'),
            'threshold_plane_error_mm':max_plane_error_mm,'max_skew_ms':max_skew_ms,'frames':records,'rejected':rejected,
            'board_sha256':digest(session/'board.json'),'frames_sha256':digest(session/'frames.csv')}
    write_json(session/f'results/depth_rgb_{model_source}_validation.json',report)
    return report


def board_pose(session,stream,frame_id):
    """Board pose for one saved image; no assumption that board==table/robot frame."""
    session=Path(session)
    row=next((r for r in rows(session,stream) if r['frame_id']==frame_id),None)
    if row is None:raise ValueError('frame-id not found for selected stream.')
    cfg=read_json(session/'board.json');board=make_board(cfg)
    model=read_json(session/f'results/{stream}_calibration.json')
    if model['board_sha256']!=digest(session/'board.json') or model['frames_sha256']!=digest(session/'frames.csv'):
        raise ValueError('Calibration is stale; rerun intrinsic after changing board/captures.')
    image=cv2.imread(str(session/row['relative_path']),cv2.IMREAD_UNCHANGED)
    obj,img,ids=detect(image,board)
    if [image.shape[1],image.shape[0]]!=model['image_size_px']:
        raise ValueError('Image size differs from calibration.')
    k,d=as_model(model)
    rms,_,rv,tv=reprojection(obj,img,k,d)
    r=cv2.Rodrigues(rv)[0];t=tv.ravel()
    if np.any((obj@r.T+t)[:,2]<=0):raise ValueError('Board pose is behind camera.')
    forward=np.eye(4);forward[:3,:3]=r;forward[:3,3]=t
    inverse=np.eye(4);inverse[:3,:3]=r.T;inverse[:3,3]=-r.T@t
    report={'status':'COMPUTED_NOT_WORLD_CALIBRATED','frame_id':frame_id,'stream':stream,
            'intrinsic_status':model['status'],'rms_px':rms,'corners':len(ids),
            'T_camera_from_board':forward.tolist(),'T_board_from_camera':inverse.tolist(),
            'translation_unit':'m','equation':'p_camera = T_camera_from_board @ p_board_homogeneous',
            'board_frame':cfg['board_frame_definition'],
            'note':'Per-image planar PnP pose; board motion changes it. Not robot/table extrinsics or hand-eye.',
            'board_sha256':digest(session/'board.json'),'frames_sha256':digest(session/'frames.csv')}
    # Use source image stem for safe, stable report filenames.
    write_json(session/f'results/{stream}_board_pose_{Path(row["relative_path"]).stem}.json',report)
    return report
