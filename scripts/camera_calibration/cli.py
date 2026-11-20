#!/usr/bin/env python3
"""Gemini 335L calibration: official/import-bag and independent offline checks."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from airbot_tt import camera_calibration as cal


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('official',help='Run an unmodified pinned Orbbec example')
    s.add_argument('example',choices=['info','parameters','profiles','record','playback'])
    s.add_argument('extra',nargs=argparse.REMAINDER)
    s=sub.add_parser('init',help='Create a new session using physically measured board dimensions')
    s.add_argument('--session',required=True,type=Path)
    s.add_argument('--board',type=Path,default=cal.ROOT/'configs/camera_board.json')
    s.add_argument('--square-mm',required=True,type=float)
    s.add_argument('--marker-mm',required=True,type=float)
    s.add_argument('--confirm-measured',required=True,action='store_true')
    s=sub.add_parser('board-check',help='Check a rendered PDF image or photo against the configured board')
    s.add_argument('--image',required=True,type=Path)
    s.add_argument('--board',type=Path,default=cal.ROOT/'configs/camera_board.json')
    s.add_argument('--output',required=True,type=Path)
    s=sub.add_parser('import-bag',help='Import raw samples from an official BAG + JSON sidecar')
    s.add_argument('--session',required=True,type=Path)
    s.add_argument('--bag',required=True,type=Path)
    s.add_argument('--stream',action='append',choices=cal.STREAMS,required=True)
    s.add_argument('--split',choices=['fit','validation'],required=True)
    s.add_argument('--count',type=int,default=20)
    s.add_argument('--interval',type=float,default=2)
    s.add_argument('--start-s',type=float,default=10)
    s.add_argument('--max-skew-ms',type=float,default=10)
    s.add_argument('--reference-z-m',type=float)
    s.add_argument('--depth-roi',type=int,nargs=4,metavar=('X','Y','WIDTH','HEIGHT'))
    s.add_argument('--frontoparallel',action='store_true')
    s=sub.add_parser('intrinsic',help='Estimate K/distortion from fit frames and check separate validation frames')
    s.add_argument('--session',required=True,type=Path)
    s.add_argument('--stream',choices=['rgb','ir_left','ir_right'],required=True)
    s.add_argument('--min-fit',type=int,default=15)
    s.add_argument('--min-validation',type=int,default=5)
    s.add_argument('--max-rms',type=float,default=.5)
    s=sub.add_parser('stereo',help='Estimate camera-to-camera extrinsics with fixed intrinsics; NOT hand-eye')
    s.add_argument('--session',required=True,type=Path)
    s.add_argument('--first',choices=['rgb','ir_left','ir_right'],required=True)
    s.add_argument('--second',choices=['rgb','ir_left','ir_right'],required=True)
    s.add_argument('--min-fit',type=int,default=15)
    s.add_argument('--min-validation',type=int,default=5)
    s.add_argument('--max-skew-ms',type=float,default=10)
    s.add_argument('--max-rms',type=float,default=1)
    s=sub.add_parser('depth-check',help='Compare raw depth to independent plane references; optional offline correction candidate')
    s.add_argument('--session',required=True,type=Path)
    s.add_argument('--max-bias-mm',type=float,default=10)
    s.add_argument('--min-valid',type=float,default=.9)
    s=sub.add_parser('alignment-check',help='Check factory depth->RGB consistency against held-out ChArUco planes')
    s.add_argument('--session',required=True,type=Path)
    s.add_argument('--max-skew-ms',type=float,default=10)
    s.add_argument('--max-plane-error-mm',type=float,default=10)
    s.add_argument('--model-source',choices=['factory','estimated'],default='factory')
    s=sub.add_parser('board-pose',help='Export camera<->board pose for a saved image; no hand-eye')
    s.add_argument('--session',required=True,type=Path)
    s.add_argument('--stream',choices=['rgb','ir_left','ir_right'],required=True)
    s.add_argument('--frame-id',required=True)
    s=sub.add_parser('factory-check',help='Validate factory intrinsics without recalibration')
    s.add_argument('--session',required=True,type=Path)
    s.add_argument('--stream',required=True,choices=['rgb','ir_left','ir_right'])
    s.add_argument('--min-validation',type=int,default=5)
    s.add_argument('--max-rms',type=float,default=.5)
    args=p.parse_args()
    if args.command=='official':
        from airbot_tt.camera_official import run_example
        extra=args.extra[1:] if args.extra[:1]==['--'] else args.extra
        return run_example(args.example,extra)
    if args.command=='init':
        cal.init_session(args.session,args.board,args.square_mm,args.marker_mm)
        print('Created:',args.session.resolve());return 0
    if args.command=='board-check':
        import cv2
        board=cal.make_board(cal.read_json(args.board),measured=False)
        image=cv2.imread(str(args.image))
        obj,corners,ids=cal.detect(image,board)
        cv2.aruco.drawDetectedCornersCharuco(image,corners.reshape(-1,1,2),ids.reshape(-1,1))
        args.output.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(args.output),image):raise IOError('Failed to write board preview.')
        print(f'Detected {len(ids)}/{len(board.getChessboardCorners())} corners; preview: {args.output}')
        return 0
    if args.command=='import-bag':
        from airbot_tt.camera_recording import import_bag
        result=import_bag(args)
    elif args.command=='intrinsic':
        if args.min_fit<3 or args.min_validation<1 or args.max_rms<=0:raise ValueError('Invalid calibration thresholds.')
        result,_=cal.intrinsic(args.session,args.stream,args.min_fit,args.min_validation,args.max_rms)
    elif args.command=='stereo':
        if args.min_fit<3 or args.min_validation<1 or args.max_rms<=0 or args.max_skew_ms<0:raise ValueError('Invalid stereo thresholds.')
        result=cal.stereo(args.session,args.first,args.second,args.min_fit,args.min_validation,args.max_skew_ms,args.max_rms)
    elif args.command=='factory-check':
        if args.min_validation<1 or args.max_rms<=0:raise ValueError('Invalid validation thresholds.')
        result=cal.factory_check(args.session,args.stream,args.min_validation,args.max_rms)
    elif args.command=='board-pose':
        result=cal.board_pose(args.session,args.stream,args.frame_id)
    elif args.command=='alignment-check':
        if args.max_skew_ms<0 or args.max_plane_error_mm<=0:raise ValueError('Invalid alignment thresholds.')
        result=cal.alignment_check(args.session,args.max_skew_ms,args.max_plane_error_mm,args.model_source)
    else:
        if args.max_bias_mm<=0 or not 0<args.min_valid<=1:raise ValueError('Invalid depth thresholds.')
        result=cal.depth_check(args.session,args.max_bias_mm,args.min_valid)
    # Full details stay in session/results; do not flood the console with per-frame reports.
    summary={k:v for k,v in result.items() if k not in ('frames','batches','validation','factory_validation','fit_frame_ids','rejected')}
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return 0 if result['status'] in ('PASS','IMPORTED','COMPUTED_NOT_WORLD_CALIBRATED') else 2


if __name__=='__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('Stopped; data already saved are retained.',file=sys.stderr);sys.exit(130)
    except (ValueError, OSError, KeyError, RuntimeError) as e:
        print(f'ERROR: {e}',file=sys.stderr);sys.exit(2)
