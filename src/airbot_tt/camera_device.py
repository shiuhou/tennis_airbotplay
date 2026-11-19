"""Orbbec SDK 2.1.3 capture adapter. Only camera streaming; no robot access."""
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time
import uuid

import cv2
import numpy as np

from .camera_calibration import FIELDS, ROOT, STREAMS, gray8, make_board, read_json, write_json


def sdk():
    logdir = ROOT/'.agents/runs/foundation/camera_calibration/sdk_logs'
    logdir.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    try:
        os.chdir(logdir)
        import pyorbbecsdk as ob
    finally:
        os.chdir(previous)
    ob.Context.set_logger_to_file(ob.OBLogLevel.ERROR, str(logdir))
    ob.Context.set_logger_to_console(ob.OBLogLevel.ERROR)
    return ob


def choose_device(ob, serial):
    context = ob.Context()
    devices = context.query_devices()
    if not devices.get_count():
        raise ValueError('No Orbbec device detected. Check USB3 cable, power and Linux udev permissions.')
    available = [devices.get_device_serial_number_by_index(i) for i in range(devices.get_count())]
    if serial not in available:
        raise ValueError(f'Requested serial not found; available serials: {available}')
    return context, devices.get_device_by_serial_number(serial)


def sensor_types(ob):
    return dict(rgb=ob.OBSensorType.COLOR_SENSOR, depth=ob.OBSensorType.DEPTH_SENSOR,
                ir_left=ob.OBSensorType.LEFT_IR_SENSOR, ir_right=ob.OBSensorType.RIGHT_IR_SENSOR)


def info(device):
    i = device.get_device_info()
    return {'model': i.get_name(), 'serial': i.get_serial_number(), 'firmware': i.get_firmware_version(),
            'sdk_package': 'pyorbbecsdk2', 'sdk_version': importlib.metadata.version('pyorbbecsdk2')}


def profile_info(profile):
    return {'width': profile.get_width(), 'height': profile.get_height(),
            'fps': profile.get_fps(), 'format': profile.get_format().name}


def inspect_devices():
    ob = sdk(); context = ob.Context(); devices = context.query_devices(); out = []
    for i in range(devices.get_count()):
        device = devices.get_device_by_index(i); item = info(device); pipeline = ob.Pipeline(device)
        item['profiles'] = {}; item['unavailable_streams'] = {}
        for name, sensor in sensor_types(ob).items():
            try:
                profiles = pipeline.get_stream_profile_list(sensor)
                item['profiles'][name] = [profile_info(profiles.get_stream_profile_by_index(j).as_video_stream_profile())
                                           for j in range(profiles.get_count())]
            except ob.OBError as e:
                item['unavailable_streams'][name] = str(e)
        out.append(item)
    return {'status': 'FOUND' if out else 'NO_DEVICE', 'devices': out}


def properties(ob, device):
    result = {}
    names = ['OB_PROP_IR_RECTIFY_BOOL', 'OB_PROP_HARDWARE_DISTORTION_SWITCH_BOOL',
             'OB_PROP_LASER_BOOL', 'OB_PROP_COLOR_AUTO_EXPOSURE_BOOL', 'OB_PROP_COLOR_EXPOSURE_INT',
             'OB_PROP_COLOR_GAIN_INT', 'OB_PROP_IR_AUTO_EXPOSURE_BOOL', 'OB_PROP_IR_EXPOSURE_INT',
             'OB_PROP_IR_GAIN_INT', 'OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL', 'OB_PROP_DEPTH_EXPOSURE_INT']
    for name in names:
        prop = getattr(ob.OBPropertyID, name)
        try:
            reader = device.get_bool_property if name.endswith('BOOL') else device.get_int_property
            result[name] = reader(prop)
        except ob.OBError:
            result[name] = None
    try:
        result['depth_preset'] = device.get_current_preset_name()
    except ob.OBError:
        result['depth_preset'] = None
    return result


def select_profiles(ob, pipeline, specs):
    profiles = {}
    for spec in specs:
        match = re.fullmatch(r'(rgb|ir_left|ir_right|depth)=(\d+)x(\d+)@(\d+):([A-Z0-9]+)', spec)
        if not match:
            raise ValueError('Profile syntax: rgb=1280x720@30:MJPG (choose a listed profile).')
        name, w, h, fps, fmt = match.groups()
        if name in profiles:
            raise ValueError('Duplicate stream selection.')
        if fmt not in ('RGB', 'BGR', 'MJPG', 'YUYV', 'UYVY', 'Y8', 'Y16'):
            raise ValueError(f'Unsupported save format {fmt}. Choose RGB/BGR/MJPG/YUYV/UYVY/Y8/Y16.')
        if name == 'depth' and fmt != 'Y16':
            raise ValueError('Depth capture requires Y16.')
        profiles[name] = pipeline.get_stream_profile_list(sensor_types(ob)[name]).get_video_stream_profile(
            int(w), int(h), getattr(ob.OBFormat, fmt), int(fps))
    if not profiles:
        raise ValueError('Select at least one stream with --profile.')
    return profiles


def export_factory(profiles):
    streams, extrinsics = {}, {}
    for name, profile in profiles.items():
        i, d = profile.get_intrinsic(), profile.get_distortion()
        streams[name] = {'profile': profile_info(profile), 'camera_matrix': [[i.fx,0,i.cx],[0,i.fy,i.cy],[0,0,1]],
                         'image_size_px': [i.width,i.height], 'distortion_model': d.model.name,
                         'distortion': {key: float(getattr(d,key)) for key in ('k1','k2','k3','k4','k5','k6','p1','p2')}}
        for other, target in profiles.items():
            if name == other:
                continue
            ext = profile.get_extrinsic_to(target)
            r = np.array(ext.rot).reshape(3,3); t = np.array(ext.transform)/1000
            if not np.all(np.isfinite(r)) or not np.all(np.isfinite(t)) or not np.allclose(r.T@r,np.eye(3),atol=1e-3) or abs(np.linalg.det(r)-1)>1e-3:
                raise ValueError('Invalid factory rigid transform.')
            extrinsics[f'{name}_to_{other}'] = {'from': name, 'to': other, 'R': r.tolist(), 't_m': t.tolist(),
                                               'equation': 'p_to = R @ p_from + t_m'}
    return {'status': 'READ_NOT_VALIDATED', 'streams': streams, 'extrinsics': extrinsics}


def decode_frame(frame):
    w, h, fmt = frame.get_width(), frame.get_height(), frame.get_format().name
    raw = np.frombuffer(frame.get_data(), dtype=np.uint8)
    if fmt == 'MJPG':
        image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    elif fmt in ('RGB','BGR'):
        image = raw.reshape(h,w,3).copy()
        if fmt == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif fmt in ('YUYV','UYVY'):
        code = cv2.COLOR_YUV2BGR_YUY2 if fmt == 'YUYV' else cv2.COLOR_YUV2BGR_UYVY
        image = cv2.cvtColor(raw.reshape(h,w,2), code)
    elif fmt == 'Y8':
        image = raw.reshape(h,w).copy()
    elif fmt == 'Y16':
        image = raw.view(np.uint16).reshape(h,w).copy()
    else:
        raise ValueError(f'Unsupported image format: {fmt}')
    if image is None or image.shape[:2] != (h,w):
        raise ValueError('Decoded frame size mismatch.')
    return image


def get_frames(frameset, names):
    getters = dict(rgb='get_color_frame',depth='get_depth_frame',ir_left='get_left_ir_frame',ir_right='get_right_ir_frame')
    return {name: getattr(frameset,getters[name])() for name in names}


def metadata(ob, frame, key):
    enum = getattr(ob.OBFrameMetadataType,key)
    return frame.get_metadata_value(enum) if frame.has_metadata(enum) else ''


def capture(args):
    session = Path(args.session)
    manifest = read_json(session/'manifest.json')
    board_cfg = read_json(session/'board.json'); board = make_board(board_cfg)
    ob = sdk(); context, device = choose_device(ob,args.serial)
    pipeline = ob.Pipeline(device); profiles = select_profiles(ob,pipeline,args.profile)
    selected = {k:profile_info(v) for k,v in profiles.items()}; device_info = info(device)
    if manifest['device'] is not None and (manifest['device'] != device_info or manifest['profiles'] != selected):
        raise ValueError('Device/firmware/SDK/profile changed. Create a new session.')
    if args.reference_z_m is not None:
        if 'depth' not in profiles or not args.depth_roi or not args.frontoparallel or args.reference_z_m <= 0:
            raise ValueError('Depth reference needs depth stream, --depth-roi, positive --reference-z-m and --frontoparallel.')
    if args.max_skew_ms < 0:
        raise ValueError('max-skew-ms must be nonnegative.')
    if args.count < 0 or args.count > 500 or args.interval <= 0 or args.warmup < 0:
        raise ValueError('count must be 0..500; interval positive; warmup nonnegative.')
    if args.headless and not args.count:
        raise ValueError('Headless capture requires a bounded --count (1..500).')
    if not args.headless and os.name != 'nt' and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        raise ValueError('No display available. Use --headless --count N, or run on a desktop.')
    cfg = ob.Config()
    for profile in profiles.values():
        cfg.enable_stream(profile)
    cfg.set_align_mode(ob.OBAlignMode.DISABLE)
    cfg.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    # No auto firmware preset/rectification/laser changes; settings are read and exported.
    bid = uuid.uuid4().hex[:12]
    batch = {'batch_id':bid,'split':args.split,'started_at':datetime.now(timezone.utc).isoformat(),
             'status':'STARTING','saved':0,'reference_z_m':args.reference_z_m,
             'depth_roi':args.depth_roi,'frontoparallel':args.frontoparallel,
             'requested_max_skew_ms':args.max_skew_ms}
    started = False
    original_laser = None
    try:
        if args.laser != 'keep':
            prop = ob.OBPropertyID.OB_PROP_LASER_BOOL
            original_laser = device.get_bool_property(prop)
            device.set_bool_property(prop, args.laser == 'on')
        pipeline.start(cfg); started = True
        end = time.monotonic()+args.warmup; last_good = time.monotonic()
        frames = None
        while frames is None or time.monotonic()<end:
            fs = pipeline.wait_for_frames(1000)
            if fs is not None:
                candidate = get_frames(fs,profiles)
                if all(f is not None for f in candidate.values()):
                    frames = candidate; last_good = time.monotonic()
            if time.monotonic()-last_good > 30:
                raise ValueError('No complete frames for 30 s. Check bandwidth/profile compatibility.')
        active_profiles = {name: frame.get_stream_profile().as_video_stream_profile() for name, frame in frames.items()}
        if {k:profile_info(v) for k,v in active_profiles.items()} != selected:
            raise ValueError('Delivered profiles differ from requested profiles.')
        factory = export_factory(active_profiles)
        settings = properties(ob,device)
        # Geometry-affecting properties cannot change within one dataset.
        old = manifest.get('device_settings',{})
        for key in ('OB_PROP_IR_RECTIFY_BOOL','OB_PROP_HARDWARE_DISTORTION_SWITCH_BOOL','depth_preset'):
            if key in old and old[key] != settings[key]:
                raise ValueError(f'{key} changed; create a new session.')
        if manifest['device'] is not None and read_json(session/'factory_calibration.json') != factory:
            raise ValueError('Factory calibration changed; use a new session.')
        manifest.update(device=device_info,profiles=selected,device_settings=settings,status='CAPTURING')
        batch['device_settings'] = settings
        manifest['batches'].append(batch)
        write_json(session/'factory_calibration.json',factory)
        write_json(session/'manifest.json',manifest)
        last_save = time.monotonic()-args.interval
        while batch['saved'] < (args.count or 500):
            fs = pipeline.wait_for_frames(1000)
            if fs is None:
                if time.monotonic()-last_good>30:
                    raise ValueError('Camera stopped delivering frames.')
                continue
            frames = get_frames(fs,profiles)
            if any(f is None for f in frames.values()):
                if time.monotonic()-last_good>30:
                    raise ValueError('Incomplete frame sets for 30 s.')
                continue
            last_good = time.monotonic()
            images = {name:decode_frame(f) for name,f in frames.items()}
            if args.depth_roi:
                if 'depth' not in images:
                    raise ValueError('--depth-roi requires a depth stream.')
                x,y,w,h=args.depth_roi
                if min(x,y)<0 or min(w,h)<=0 or x+w>images['depth'].shape[1] or y+h>images['depth'].shape[0]:
                    raise ValueError('Depth ROI outside original depth image.')
            timestamps = [f.get_timestamp_us() for f in frames.values()]
            skew_ms = (max(timestamps)-min(timestamps))/1000
            key = -1
            if not args.headless:
                for name,image in images.items():
                    preview = cv2.cvtColor(gray8(image),cv2.COLOR_GRAY2BGR) if image.ndim==2 else image.copy()
                    if name != 'depth':
                        c,ids,_,_ = cv2.aruco.CharucoDetector(board).detectBoard(gray8(image))
                        if ids is not None:
                            cv2.aruco.drawDetectedCornersCharuco(preview,c,ids)
                    if name=='depth' and args.depth_roi:
                        x,y,w,h=args.depth_roi; cv2.rectangle(preview,(x,y),(x+w,y+h),(0,255,0),2)
                    scale=min(1,960/preview.shape[1]);preview=cv2.resize(preview,None,fx=scale,fy=scale)
                    cv2.putText(preview,f'{name} {args.split} saved={batch["saved"]} skew={skew_ms:.1f}ms S:save Q:quit',
                                (8,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
                    cv2.imshow(name,preview)
                key=cv2.waitKey(1)&255
                if key in (27,ord('q')):
                    break
            save = (args.count and time.monotonic()-last_save>=args.interval) or key==ord('s')
            if not save:
                continue
            if skew_ms > args.max_skew_ms:
                raise ValueError(f'Frame timestamp skew {skew_ms:.2f}ms exceeds {args.max_skew_ms}ms. Board must be static; check profiles/sync.')
            frame_id=f'{bid}_{batch["saved"]:04d}'; records=[]; saved_paths=[]
            host_stamp=time.monotonic_ns()
            try:
                for name,image in images.items():
                    dest=session/name/f'{frame_id}.png'
                    if dest.exists():
                        raise ValueError('Capture path collision.')
                    if not cv2.imwrite(str(dest),image):
                        raise IOError(f'Could not save {dest}')
                    saved_paths.append(dest)
                    frame=frames[name]
                    scale=float(frame.get_depth_scale())/1000 if name=='depth' else ''
                    if name=='depth' and (not np.isfinite(scale) or scale<=0):
                        raise ValueError('Invalid SDK depth scale.')
                    records.append(dict(frame_id=frame_id,batch_id=bid,stream=name,relative_path=str(dest.relative_to(session)),
                         device_timestamp_us=frame.get_timestamp_us(),host_timestamp_ns=host_stamp,
                         exposure_sdk_raw=metadata(ob,frame,'EXPOSURE'),gain_sdk_raw=metadata(ob,frame,'GAIN'),split=args.split,
                         depth_scale_m_per_unit=scale,reference_z_m=args.reference_z_m if name=='depth' else '',
                         roi_xywh=json.dumps(args.depth_roi) if name=='depth' and args.depth_roi else '',
                         frontoparallel=args.frontoparallel if name=='depth' else ''))
                with (session/'frames.csv').open('a',newline='') as f:
                    csv.DictWriter(f,fieldnames=FIELDS).writerows(records);f.flush();os.fsync(f.fileno())
            except Exception:
                # Partial files are left for inspection if CSV writing itself failed.
                batch['partial_files']=[str(p.relative_to(session)) for p in saved_paths]
                raise
            batch['saved']+=1;last_save=time.monotonic()
            write_json(session/'manifest.json',manifest)
            print(f'Saved {frame_id} ({args.split}), skew={skew_ms:.2f} ms',flush=True)
        batch['status']='CAPTURED' if batch['saved'] else 'NO_FRAMES_SAVED'
    except BaseException as e:
        batch.update(status='INTERRUPTED' if isinstance(e,KeyboardInterrupt) else 'ERROR',error=str(e))
        raise
    finally:
        if started:
            pipeline.stop()
        if original_laser is not None:
            device.set_bool_property(ob.OBPropertyID.OB_PROP_LASER_BOOL, original_laser)
        if not args.headless:
            cv2.destroyAllWindows()
        if batch in manifest['batches']:
            manifest['status']='CAPTURED_NOT_CALIBRATED' if batch['status']=='CAPTURED' else batch['status']
            write_json(session/'manifest.json',manifest)
    return batch
