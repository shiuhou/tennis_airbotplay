"""Independent synthetic geometry checks; run in the camera environment."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
try:
    import cv2
    from airbot_tt import camera_calibration as cal
    CAMERA_AVAILABLE = hasattr(cv2,'aruco') and hasattr(cv2.aruco,'CharucoDetector')
except ImportError:
    CAMERA_AVAILABLE = False


@unittest.skipUnless(CAMERA_AVAILABLE, 'Use scripts/camera_calibration/setup.sh for camera tests')
class CameraCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.session = Path(self.tmp.name)/'session'
        cal.init_session(self.session,cal.ROOT/'configs/camera_board.json',15,11)
        self.board=cal.make_board(cal.read_json(self.session/'board.json'))

    def synthetic_views(self):
        k=np.array([[900.,0,640],[0,920.,480],[0,0,1.]])
        size=(1280,960)
        r_ab=cv2.Rodrigues(np.array([.01,-.025,.005]))[0]
        t_ab=np.array([.045,.002,.001])
        drawing=self.board.generateImage((1100,800),marginSize=0)
        src=np.float32([[0,0],[1099,0],[1099,799],[0,799]])
        outer=np.float32([[0,0,0],[.165,0,0],[.165,.12,0],[0,.12,0]])
        rng=np.random.default_rng(91)
        records=[]
        for i in range(24):
            rv=np.array([rng.uniform(-.55,.55),rng.uniform(-.6,.6),rng.uniform(-.2,.2)])
            r=cv2.Rodrigues(rv)[0]
            tv=np.array([rng.uniform(-.075,.065),rng.uniform(-.05,.05),rng.uniform(.33,.52)])-r@np.array([.0825,.06,0])
            for stream,rotation,translation in [('rgb',r,tv),('ir_left',r_ab@r,r_ab@tv+t_ab)]:
                pts=cv2.projectPoints(outer,cv2.Rodrigues(rotation)[0],translation,k,np.zeros(5))[0].reshape(-1,2)
                h=cv2.getPerspectiveTransform(src,pts)
                image=cv2.warpPerspective(drawing,h,size,borderValue=255)
                path=self.session/stream/f'{i:03d}.png'
                self.assertTrue(cv2.imwrite(str(path),image))
                records.append(dict(frame_id=str(i),batch_id='synthetic',stream=stream,relative_path=str(path.relative_to(self.session)),
                                    device_timestamp_us=i*1000000,host_timestamp_ns=i*1000000000,split='fit' if i<18 else 'validation'))
        with (self.session/'frames.csv').open('a',newline='') as f:
            csv.DictWriter(f,fieldnames=cal.FIELDS).writerows(records)
        return k,r_ab,t_ab

    def test_render_detect_intrinsics_and_stereo_direction(self):
        k,r,t=self.synthetic_views()
        for stream in ['rgb','ir_left']:
            model,report=cal.intrinsic(self.session,stream)
            np.testing.assert_allclose(np.array(model['camera_matrix'])[[0,1],[0,1]],k[[0,1],[0,1]],rtol=.025)
            self.assertLess(report['estimated']['rms_px'],.5)
            self.assertEqual(len(report['validation_frame_ids']),6)
        pose=cal.board_pose(self.session,'rgb','20')
        np.testing.assert_allclose(np.array(pose['T_camera_from_board'])@np.array(pose['T_board_from_camera']),np.eye(4),atol=1e-9)
        self.assertLess(pose['rms_px'],.5)
        stereo=cal.stereo(self.session,'rgb','ir_left')
        np.testing.assert_allclose(stereo['t_m'],t,atol=.003)
        np.testing.assert_allclose(stereo['R'],r,atol=.012)
        self.assertLess(stereo['validation']['rms_px'],1)

    def test_factory_validation_does_not_fit_new_intrinsics(self):
        k,_,_=self.synthetic_views()
        cal.write_json(self.session/'factory_calibration.json',{'streams':{'rgb':{
            'image_size_px':[1280,960],'camera_matrix':k.tolist(),'distortion_model':'NONE'}}})
        report=cal.factory_check(self.session,'rgb')
        self.assertLess(report['rms_px'],.5)
        self.assertEqual(len(report['per_frame']),6)
        self.assertFalse((self.session/'results/rgb_calibration.json').exists())
        self.assertTrue((self.session/'results/rgb_factory_validation.json').exists())

    def test_board_pattern_and_unmeasured_rejection(self):
        cfg=cal.read_json(cal.ROOT/'configs/camera_board.json')
        with self.assertRaisesRegex(ValueError,'measured'):
            cal.make_board(cfg)
        img=self.board.generateImage((1200,900),marginSize=30)
        _,_,ids=cal.detect(img,self.board)
        self.assertEqual(len(ids),70)
        with self.assertRaisesRegex(ValueError,'already exists'):
            cal.init_session(self.session,cal.ROOT/'configs/camera_board.json',15,11)

    def test_validation_cannot_reuse_identical_fit_image(self):
        image=self.board.generateImage((1200,900),marginSize=30)
        path=self.session/'rgb'/'same.png';cv2.imwrite(str(path),image)
        with (self.session/'frames.csv').open('a',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=cal.FIELDS)
            for i,split in enumerate(('fit','validation')):
                writer.writerow(dict(frame_id=str(i),stream='rgb',relative_path='rgb/same.png',split=split))
        with self.assertRaisesRegex(ValueError,'Identical image'):
            cal.observations(self.session,'rgb',self.board)

    def test_factory_model_order_and_unsupported_inverse(self):
        rec={'camera_matrix':[[500,0,320],[0,500,240],[0,0,1]],'distortion_model':'BROWN_CONRADY_K6',
             'distortion':dict(k1=1,k2=2,k3=3,k4=4,k5=5,k6=6,p1=7,p2=8)}
        _,d=cal.factory_model(rec)
        np.testing.assert_array_equal(d,[1,2,7,8,3,4,5,6])
        rec['distortion_model']='INVERSE_BROWN_CONRADY'
        with self.assertRaisesRegex(ValueError,'not supported'):
            cal.factory_model(rec)

    def test_depth_units_invalid_pixels_and_roi(self):
        raw=np.full((20,30),2000,dtype=np.uint16);raw[0,:]=0
        s=cal.depth_statistics(raw,.0005,[0,0,30,20],.99)
        self.assertAlmostEqual(s['bias_m'],.01)
        self.assertAlmostEqual(s['valid_fraction'],.95)
        with self.assertRaisesRegex(ValueError,'outside'):
            cal.depth_statistics(raw,.0005,[20,0,20,20],1)
        with self.assertRaisesRegex(ValueError,'30 valid'):
            cal.depth_statistics(np.zeros_like(raw),.0005,[0,0,30,20],1)

    def test_depth_candidate_uses_only_fit_and_reports_validation(self):
        rec=[]
        for i,reference in enumerate([.5,.8,1.1,.7]):
            raw_value=round((reference-.02)/1.02*10000)
            path=self.session/'depth'/f'{i}.png'
            cv2.imwrite(str(path),np.full((20,20),raw_value,np.uint16))
            rec.append(dict(frame_id=str(i),batch_id=str(i),stream='depth',relative_path=str(path.relative_to(self.session)),
                            split='fit' if i<3 else 'validation',depth_scale_m_per_unit=.0001,reference_z_m=reference,
                            roi_xywh='[0,0,20,20]',frontoparallel=True))
        with (self.session/'frames.csv').open('a',newline='') as f:csv.DictWriter(f,fieldnames=cal.FIELDS).writerows(rec)
        report=cal.depth_check(self.session)
        self.assertEqual(report['status'],'REVIEW_REQUIRED')
        self.assertAlmostEqual(report['correction']['scale'],1.02,places=3)
        self.assertLess(abs(report['correction']['validation'][0]['corrected_bias_m']),.0002)
        self.assertIsNone(report['batches'][0]['temporal_std_of_roi_medians_m'])

    def test_depth_rgb_plane_transform_and_scale(self):
        # A known plane at Z=1 in RGB, with depth origin translated +0.1 m in RGB.
        raw=np.full((120,160),900,np.uint16)
        k=np.array([[500.,0,80],[0,500.,60],[0,0,1]])
        args=(k,np.zeros(5),np.eye(3),[0,0,.1],np.eye(3),[-.15,-.12,1.],(.3,.24))
        stats=cal.plane_residuals(raw,.001,*args,step=1)
        self.assertLess(stats['plane_p95_absolute_error_m'],1e-10)
        wrong=cal.plane_residuals(raw,.001,k,np.zeros(5),np.eye(3),[0,0,-.1],np.eye(3),[-.15,-.12,1.],(.3,.24),step=1)
        self.assertAlmostEqual(wrong['plane_median_signed_error_m'],-.2)

    def test_capture_decode_preserves_depth_bits_and_rgb_order(self):
        from airbot_tt.camera_device import decode_frame
        class Frame:
            def __init__(self, data, fmt):self.data=data;self.fmt=fmt
            def get_width(self):return 2
            def get_height(self):return 1
            def get_format(self):return type('Format',(),{'name':self.fmt})()
            def get_data(self):return self.data
        raw=np.array([1234,65535],np.uint16)
        np.testing.assert_array_equal(decode_frame(Frame(raw.tobytes(),'Y16')),raw.reshape(1,2))
        rgb=np.array([[[255,0,0],[0,255,0]]],np.uint8)
        np.testing.assert_array_equal(decode_frame(Frame(rgb.tobytes(),'RGB')),rgb[:,:,::-1])
        with self.assertRaises(ValueError):decode_frame(Frame(b'wrong','Y16'))

    def test_pairing_rejects_unsynchronized_or_mixed_split(self):
        obj=self.board.getChessboardCorners()[:20]
        base=dict(obj=obj,img=np.zeros((20,2),np.float32),ids=np.arange(20),row=dict(frame_id='a',split='fit',device_timestamp_us='1000'))
        other={**base,'row':dict(frame_id='a',split='fit',device_timestamp_us='50000')}
        pairs,rejected=cal.pair_views([base],[other],10)
        self.assertFalse(pairs);self.assertIn('skew',rejected[0]['reason'])
        other['row'].update(split='validation',device_timestamp_us='1000')
        pairs,rejected=cal.pair_views([base],[other],10)
        self.assertFalse(pairs);self.assertIn('split',rejected[0]['reason'])


if __name__=='__main__':unittest.main()
