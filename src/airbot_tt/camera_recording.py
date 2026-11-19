"""Offline bridge from official Orbbec BAG recordings to calibration datasets.

Acquisition/preview/enumeration belong to unmodified official examples. This
module only reads PlaybackDevice and never opens or configures live hardware.
"""
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import queue
import time

import cv2
import numpy as np
from .camera_calibration import FIELDS, ROOT, STREAMS, digest, read_json, write_json
from .camera_official import load_module, specification


def load_playback():
    work = ROOT/'.agents/runs/foundation/camera_calibration/official_logs'
    work.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    try:
        os.chdir(work)
        return load_module('examples/advanced/02_playback.py', 'orbbec_official_playback')
    finally:
        os.chdir(previous)


def profile_info(profile):
    return {'width': profile.get_width(), 'height': profile.get_height(),
            'fps': profile.get_fps(), 'format': profile.get_format().name}


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


def raw_image(frame, name, official_color):
    w,h=frame.get_width(),frame.get_height()
    fmt=frame.get_format().name
    raw=np.frombuffer(frame.get_data(),np.uint8)
    if name=='rgb':
        # Upstream uses np.resize; reject malformed buffers rather than repeating bytes.
        expected={'RGB':w*h*3,'YUYV':w*h*2,'UYVY':w*h*2}
        if fmt not in ('RGB','MJPG','YUYV','UYVY'):
            raise ValueError(f'Unsupported upstream color conversion: {fmt}; record RGB/MJPG/YUYV/UYVY.')
        if fmt in expected and len(raw)!=expected[fmt]:raise ValueError('Color buffer size mismatch.')
        image=official_color(frame)
    elif fmt=='Y16':
        image=raw.view(np.uint16).reshape(h,w).copy()
    elif name!='depth' and fmt=='Y8':
        image=raw.reshape(h,w).copy()
    elif name!='depth' and fmt=='MJPG':
        image=cv2.imdecode(raw,cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError(f'Unsupported {name} format {fmt}; depth requires Y16.')
    if image is None or image.shape[:2]!=(h,w):raise ValueError('Decoded dimensions do not match profile.')
    return image


def import_bag(args):
    session=Path(args.session)
    bag=Path(args.bag).resolve(); sidecar=bag.with_suffix('.json')
    if not bag.is_file() or not sidecar.is_file():
        raise ValueError('Official recording .bag AND matching .json sidecar are required.')
    if not 1<=args.count<=500 or args.interval<=0 or args.start_s<0 or args.max_skew_ms<0:
        raise ValueError('Invalid sampling count/interval/start/skew.')
    if len(set(args.stream))!=len(args.stream):raise ValueError('Duplicate stream selection.')
    if args.reference_z_m is not None and (args.reference_z_m<=0 or not args.depth_roi or not args.frontoparallel or 'depth' not in args.stream):
        raise ValueError('Depth reference needs a positive Z, depth stream, ROI and frontoparallel declaration.')
    manifest=read_json(session/'manifest.json')
    bag_hash=digest(bag)
    if any(b.get('source_sha256')==bag_hash for b in manifest['batches']):
        raise ValueError('This recording was already imported. Use a separate recording for independent validation.')
    source=load_playback();ob=source.ob
    playback=ob.PlaybackDevice(str(bag))
    if not source.load_sidecar_json(playback,str(bag)):
        raise ValueError('Official sidecar loading failed; refusing ambiguous replay configuration.')
    pipeline, config=source.setup_camera(playback)
    # Use upstream-restored recorded profiles, selecting only the requested dataset streams.
    names={'COLOR_STREAM':'rgb','DEPTH_STREAM':'depth','LEFT_IR_STREAM':'ir_left','RIGHT_IR_STREAM':'ir_right'}
    profiles=config.get_enabled_stream_profile_list();selected={}
    for i in range(profiles.get_count()):
        p=profiles.get_stream_profile_by_index(i)
        name=names.get(p.get_type().name)
        if name in args.stream:selected[name]=p.as_video_stream_profile()
    if set(selected)!=set(args.stream):raise ValueError(f'Recorded profiles missing requested streams: {set(args.stream)-set(selected)}')
    config.disable_all_stream()
    for p in selected.values():config.enable_stream(p)
    config.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    info=playback.get_device_info()
    identity={'model':info.get_name(),'serial':info.get_serial_number(),'firmware':info.get_firmware_version()}
    profile_map={n:profile_info(p) for n,p in selected.items()}
    if manifest['device'] is not None and (manifest['device']!=identity or manifest['profiles']!=profile_map):
        raise ValueError('Recorded device/firmware/profiles changed; create a new session.')
    factory=export_factory(selected)
    if manifest['device'] is not None and read_json(session/'factory_calibration.json')!=factory:
        raise ValueError('Factory calibration differs between recordings; use a separate session.')
    getters={'rgb':'get_color_frame','depth':'get_depth_frame','ir_left':'get_left_ir_frame','ir_right':'get_right_ir_frame'}
    batch={'batch_id':bag_hash[:12],'source_bag':str(bag),'source_sha256':bag_hash,
           'sidecar_sha256':digest(sidecar),'split':args.split,'saved':0,'status':'IMPORTING',
           'start_s':args.start_s,'interval_s':args.interval,'requested_count':args.count,
           'timestamp_skew_rejected':0,'queue_overflow':0,
           'sdk_version':importlib.metadata.version('pyorbbecsdk2'),
           'official_commit':specification()['commit'],'imported_at':datetime.now(timezone.utc).isoformat(),
           'reference_z_m':args.reference_z_m,'roi_xywh':args.depth_roi,'frontoparallel':args.frontoparallel}
    processing={'source':'official BAG + sidecar','alignment':'no extra alignment applied',
                'filters':'raw callback; upstream preview recommended filters not reapplied',
                'undistortion_map':{k.name:v for k,v in source.state.undistortion_map.items()}}
    if manifest['device'] is not None:
        if manifest.get('processing') != processing or any(
                b.get('sdk_version') != batch['sdk_version'] for b in manifest['batches']):
            raise ValueError('Recorded processing or SDK changed; create a new session.')
    manifest.update(device=identity,profiles=profile_map,status='IMPORTING',processing=processing)
    manifest['batches'].append(batch)
    write_json(session/'factory_calibration.json',factory)
    write_json(session/'manifest.json',manifest)
    pending=queue.Queue(maxsize=16)
    def callback(fs):
        try:pending.put_nowait(fs)
        except queue.Full:batch['queue_overflow']+=1
    started=False; first_stamp=None; target=args.start_s; last=time.monotonic()
    try:
        pipeline.start(config,callback);started=True
        while batch['saved']<args.count:
            try:fs=pending.get(timeout=1)
            except queue.Empty:
                if playback.get_playback_status()==ob.OBPlaybackStatus.STOPPED:break
                if time.monotonic()-last>30:raise ValueError('Replay delivered no complete frames for 30 s.')
                continue
            last=time.monotonic()
            frames={name:getattr(fs,getters[name])() for name in args.stream}
            if any(f is None for f in frames.values()):continue
            stamps=[f.get_timestamp_us() for f in frames.values()]
            if first_stamp is None:first_stamp=min(stamps)
            elapsed=(min(stamps)-first_stamp)/1e6
            if elapsed<target:continue
            if (max(stamps)-min(stamps))/1000>args.max_skew_ms:
                batch['timestamp_skew_rejected']+=1;continue
            records=[];fid=f'{batch["batch_id"]}_{batch["saved"]:04d}'
            for name,frame in frames.items():
                if profile_info(frame.get_stream_profile().as_video_stream_profile())!=profile_map[name]:
                    raise ValueError('Delivered playback profile changed.')
                image=raw_image(frame,name,source.frame_to_bgr_image)
                if name=='depth' and args.depth_roi:
                    x,y,w,h=args.depth_roi
                    if min(x,y)<0 or min(w,h)<=0 or x+w>image.shape[1] or y+h>image.shape[0]:
                        raise ValueError('ROI outside recorded depth dimensions.')
                destination=session/name/f'{fid}.png'
                if destination.exists():raise ValueError('Import would overwrite existing images.')
                if not cv2.imwrite(str(destination),image):raise IOError('Image save failed.')
                scale=float(frame.get_depth_scale())/1000 if name=='depth' else ''
                if name=='depth' and (not np.isfinite(scale) or scale<=0):raise ValueError('Invalid depth scale.')
                def metadata(key):
                    kind=getattr(ob.OBFrameMetadataType,key)
                    return frame.get_metadata_value(kind) if frame.has_metadata(kind) else ''
                records.append(dict(frame_id=fid,batch_id=batch['batch_id'],stream=name,
                    relative_path=str(destination.relative_to(session)),device_timestamp_us=frame.get_timestamp_us(),
                    host_timestamp_ns='',exposure_sdk_raw=metadata('EXPOSURE'),gain_sdk_raw=metadata('GAIN'),
                    split=args.split,depth_scale_m_per_unit=scale,
                    reference_z_m=args.reference_z_m if name=='depth' else '',
                    roi_xywh=json.dumps(args.depth_roi) if name=='depth' and args.depth_roi else '',
                    frontoparallel=args.frontoparallel if name=='depth' else ''))
            with (session/'frames.csv').open('a',newline='') as f:
                csv.DictWriter(f,fieldnames=FIELDS).writerows(records);f.flush();os.fsync(f.fileno())
            batch['saved']+=1;target=elapsed+args.interval
        batch['status']='IMPORTED' if batch['saved']==args.count else 'PARTIAL'
        if batch['queue_overflow'] or batch['timestamp_skew_rejected']:batch['status']='REVIEW_REQUIRED'
    except BaseException as e:
        batch.update(status='ERROR',error=str(e));raise
    finally:
        try:
            if started:pipeline.stop()
        finally:
            manifest['status']=batch['status']
            write_json(session/'manifest.json',manifest)
    return batch
