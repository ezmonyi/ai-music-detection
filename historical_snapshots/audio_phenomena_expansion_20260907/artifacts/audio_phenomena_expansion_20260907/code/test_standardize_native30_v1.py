import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import standardize_native30_v1 as m


class Native30Tests(unittest.TestCase):
    def test_integer_center_and_odd_remainder(self):
        c=m.center_coordinates(30*48000+3,48000)
        self.assertEqual(c['crop_start_frame'],1)
        self.assertEqual(c['crop_frames'],1440000)

    def test_short_no_padding(self):
        with self.assertRaises(ValueError):m.center_coordinates(30*44100-1,44100)

    def test_region_bounds_and_native_hash_required(self):
        for r in [{'start_frame':0,'frames':60},
                  {'start_frame':0,'frames':60,'float64_sha256':'bad'},
                  {'start_frame':1,'frames':60,'float64_sha256':'a'*64}]:
            with self.assertRaises(ValueError):m.center_coordinates(60,1,r)

    def test_region_center_not_full_source_center(self):
        c=m.center_coordinates(1000,10,{'start_frame':0,'frames':600,'float64_sha256':'a'*64})
        self.assertEqual(c['crop_start_frame'],150)
        self.assertNotEqual(c['crop_start_frame'],350)

    def test_invalid_counts(self):
        for f,r in [(True,44100),(30,True),(30.0,1),(30,0),(-1,1)]:
            with self.assertRaises(ValueError):m.center_coordinates(f,r)

    def test_resample_no_gain_or_channel_mixing(self):
        y=np.zeros((m.FRAMES,2),np.float64);y[7]=[2,-3]
        out=m.resample_crop(y,44100)
        self.assertEqual(out.dtype,np.dtype('<f4'))
        np.testing.assert_array_equal(out,y.astype('<f4'))

    def test_real_resampling_preserves_exact_length(self):
        t=np.arange(30*48000)/48000
        y=np.column_stack([np.sin(2*np.pi*100*t),.2*np.cos(2*np.pi*700*t)])
        out=m.resample_crop(y,48000)
        self.assertEqual(out.shape,(1323000,2));self.assertTrue(np.isfinite(out).all())
        self.assertGreater(float(np.max(np.abs(out[:,0]))),.9)

    def test_resampling_validation(self):
        for y in [np.zeros((m.FRAMES,1),np.float64),np.zeros((m.FRAMES,2),np.float32),np.full((m.FRAMES,2),np.nan)]:
            with self.assertRaises(ValueError):m.resample_crop(y,44100)

    def fixture(self,root):
        path=Path(root)/'native.wav'; t=np.arange(31*100)/100
        y=np.column_stack([.2+np.sin(2*np.pi*3*t),2*np.cos(2*np.pi*7*t)])
        sf.write(path,y,100,subtype='DOUBLE')
        return path,y,m.digest(path)

    def test_full_sequential_end_to_end(self):
        with tempfile.TemporaryDirectory() as root:
            path,y,sha=self.fixture(root);out,a=m.standardize(path,sha,100,2)
            self.assertEqual(out.shape,(m.FRAMES,2));self.assertEqual(a['coordinates']['crop_start_frame'],50)
            self.assertEqual(a['native_crop_float64_sha256'],m.pcm_hash(y[50:3050]))
            self.assertEqual(a['sequential_decode']['actual_frames'],3100)
            self.assertEqual(a['sequential_decode']['read_calls_including_empty_eof'],2)
            self.assertGreater(a['samples_abs_above_one'],0)
            self.assertEqual(sha,m.digest(path));self.assertFalse(a['classifier_admitted'])

    def test_approved_region_replayed_before_crop(self):
        with tempfile.TemporaryDirectory() as root:
            path,y,sha=self.fixture(root);region={'start_frame':0,'frames':3000,'float64_sha256':m.pcm_hash(y[:3000])}
            out,a=m.standardize(path,sha,100,2,approved_region=region)
            self.assertEqual(a['coordinates']['crop_start_frame'],0)
            self.assertEqual(a['region_kind'],'approved_native_interval')
            region['float64_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'region hash'):m.standardize(path,sha,100,2,approved_region=region)

    def test_provenance_sha_channels_rate_rejections(self):
        with tempfile.TemporaryDirectory() as root:
            path,_,sha=self.fixture(root)
            for digest,rate,ch in [('0'*64,100,2),(sha,100,1),(sha,101,2),(sha,100,True)]:
                with self.assertRaises(ValueError):m.standardize(path,digest,rate,ch)

    def test_source_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path,_,sha=self.fixture(root)
            with patch.object(m,'signature',side_effect=[(1,2,3,4,5),(1,2,3,4,6)]):
                with self.assertRaisesRegex(ValueError,'source changed'):m.standardize(path,sha,100,2)

    def test_second_pass_difference_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path,_,sha=self.fixture(root);original=m.sequential;calls=[]
            def wrong(*args,**kwargs):
                y,a=original(*args,**kwargs);calls.append(1)
                if len(calls)==2:a['float64_pcm_sha256']='0'*64
                return y,a
            with patch.object(m,'sequential',side_effect=wrong):
                with self.assertRaisesRegex(ValueError,'decode changed'):m.standardize(path,sha,100,2)

    def test_nonfinite_native_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'nonfinite.wav'; y=np.zeros((3100,2));y[0,0]=np.nan;sf.write(path,y,100,subtype='DOUBLE')
            with self.assertRaisesRegex(ValueError,'nonfinite'):m.standardize(path,m.digest(path),100,2)

    def test_capture_spans_many_blocks(self):
        with tempfile.TemporaryDirectory() as root:
            path,y,sha=self.fixture(root)
            with patch.object(m,'BLOCK',97):
                captured,a=m.sequential(path,100,(51,3002))
            np.testing.assert_array_equal(captured,y[51:3002])
            self.assertGreater(a['read_calls_including_empty_eof'],30)
            self.assertEqual(a['actual_frames'],len(y))

    def test_short_nonempty_read_is_not_eof(self):
        data=np.arange(66,dtype=np.float64).reshape(33,2)
        class ShortReader:
            samplerate=100;channels=2;frames=99;format='MOCK';subtype='DOUBLE'
            def __init__(self,*args):self.position=0
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def read(self,n,**kwargs):
                end=min(self.position+7,len(data));r=data[self.position:end];self.position=end;return r
        with patch.object(m.sf,'SoundFile',ShortReader):
            captured,a=m.sequential('synthetic-no-file',100,(3,31))
        np.testing.assert_array_equal(captured,data[3:31])
        self.assertEqual(a['actual_frames'],33)
        self.assertEqual(a['header']['frames'],99)
        self.assertEqual(a['read_calls_including_empty_eof'],6)


if __name__=='__main__':unittest.main()
