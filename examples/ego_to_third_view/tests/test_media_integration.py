"""Real FFmpeg CPU integration. Synthetic media does not establish cross-scene reconstruction."""
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from automation.contracts import digest
from automation.media import prepare_reference,probe,retime,review_packet,run_ffmpeg,same_clock,threeway
from automation.timeline import pchip_map


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg/ffprobe external media runtime not installed')
class MediaIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.source=self.root/'source.mp4'
        run_ffmpeg(['-f','lavfi','-i','testsrc2=size=96x64:rate=24:duration=5','-c:v','libx264',
                    '-pix_fmt','yuv420p',self.source])

    def tearDown(self):self.tmp.cleanup()

    def test_reference_pads_without_truncating_or_modifying_source(self):
        before=digest(self.source);out=self.root/'reference.mp4'
        report=prepare_reference(self.source,out)
        self.assertEqual(report['reference_frames'],124);self.assertEqual(probe(out)['frames'],124)
        self.assertEqual(digest(self.source),before)

    def test_monotone_original_frame_selection_and_clock(self):
        mapping=pchip_map([[0,0],[30,24],[90,83],[119,119]],120)
        output=self.root/'retimed.mp4';retime(self.source,output,mapping)
        self.assertTrue(same_clock(probe(self.source),probe(output)))
        try:import cv2;import numpy as np
        except ImportError:self.skipTest('OpenCV needed for independent pixel correspondence check')
        cap=cv2.VideoCapture(str(self.source));frames=[]
        while True:
            ok,frame=cap.read()
            if not ok:break
            frames.append(frame)
        cap.release();cap=cv2.VideoCapture(str(output));errors=[]
        for index in mapping['source_frames']:
            ok,frame=cap.read();self.assertTrue(ok)
            errors.append(float(np.abs(frame.astype(float)-frames[index]).mean()))
        cap.release();self.assertLess(max(errors),8.)

    def test_review_handles_portrait_and_partial_last_sheet(self):
        source=self.root/'portrait.mp4'
        run_ffmpeg(['-f','lavfi','-i','testsrc2=size=64x96:rate=24:duration=1.3','-c:v','libx264','-pix_fmt','yuv420p',source])
        report=review_packet({'portrait':source},self.root/'review')
        self.assertEqual(len(list((self.root/'review').glob('portrait_*.jpg'))),2)
        self.assertEqual(report['videos']['portrait']['frames'],32)

    def test_threeway_clock_and_source_hashes(self):
        before=digest(self.source);out=self.root/'threeway.mp4'
        threeway(self.source,self.source,self.source,out)
        self.assertEqual(probe(out)['frames'],120);self.assertEqual(digest(self.source),before)

    def test_output_overwrite_rejected(self):
        with self.assertRaises(Exception):prepare_reference(self.source,self.source)


if __name__=='__main__':unittest.main()
