import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from automation.contracts import (INVARIANTS,digest,read_json,validate_manifest,
                                  validate_review,validate_scene,write_json)
from automation.gpu import Lease,select
from automation.h3_worker import ResidentGenerator,payload_for
from automation.timeline import event_anchors,pchip_map,reference_frame_count


def scene():
    return {'schema_version':1,'source_sha256':'0'*64,
        'coordinate_system':{'frame':'scene_world','unit':'m','up':'+z','handedness':'right'},
        'actor':{'origin_authority':'source_ego_world','origin_evidence':'source-view calibrated operator side',
                 'origin_confidence':.9,'root_m':[0,0,0],'heading_rad':0.,'upper_arm_m':.36,
                 'forearm_m':.32,'shoulder_width_m':.36,'length_policy':'constant_per_actor'},
        'render_camera':{'eye_m':[1,-1,1],'target_m':[0,0,0]},
        'objects':[{'id':'arbitrary-mug','mesh':'mug.obj','description':'ceramic mug'}]}


def review(binding,phase='pre',decision='accept',issues=None):
    return {'phase':phase,'binding':binding,'reviewer':{'kind':'human','name':'test-only simulated reviewer'},
            'decision':decision,'inspected_frames':{'ranges':[[0,binding['frames']-1]]},
            'checks':{k:'pass' for k in ['actor_origin','limb_lengths','torso','contacts','identity','occlusion','camera','timing','background']},
            'issues':issues or []}


class ContractTests(unittest.TestCase):
    def test_camera_cannot_author_actor(self):
        value=scene();value['actor']['origin_authority']='render_camera'
        with self.assertRaisesRegex(ValueError,'Actor placement'):validate_scene(value)

    def test_unsupported_locomotion_is_explicit(self):
        value=scene();value['actor']['root_motion']='walking'
        with self.assertRaisesRegex(ValueError,'locomotion'):validate_scene(value)

    def test_uncertain_scale_origin_not_silently_defaulted(self):
        value=scene();value['actor']['origin_confidence']=.3
        with self.assertRaises(ValueError):validate_scene(value)

    def test_object_symmetry_not_inferred_from_name(self):
        value=scene();value['objects'][0]['orientation_mode']='world_locked'
        with self.assertRaisesRegex(ValueError,'symmetry'):validate_scene(value)

    def test_each_hard_invariant_cannot_be_waived(self):
        for key,value in INVARIANTS.items():
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_manifest({'schema_version':1,'clips':[{'id':'a','source':'/video'}],
                                   'invariants':{key:not value if isinstance(value,bool) else 'wrong'}})

    def test_duplicate_clip_and_path_injection_rejected(self):
        for clips in [[{'id':'../escape','source':'/video'}],[{'id':'a','source':'/a'},{'id':'a','source':'/b'}]]:
            with self.assertRaises(ValueError):validate_manifest({'schema_version':1,'clips':clips})

    def test_review_cannot_accept_unknown(self):
        binding={'frames':120};value=review(binding);value['checks']['contacts']='unknown'
        with self.assertRaises(ValueError):validate_review(value,binding,'pre')

    def test_review_must_cover_whole_timeline(self):
        binding={'frames':120};value=review(binding);value['inspected_frames']=[0,60,119]
        with self.assertRaisesRegex(ValueError,'full-timeline'):validate_review(value,binding,'pre')

    def test_stale_review_rejected(self):
        value=review({'frames':120,'sha':'old'})
        with self.assertRaisesRegex(ValueError,'stale'):validate_review(value,{'frames':120,'sha':'new'},'pre')

    def test_full_review_accepts(self):
        binding={'frames':120};self.assertEqual(validate_review(review(binding),binding,'pre')['decision'],'accept')


class TimelineTests(unittest.TestCase):
    def test_reference_never_floors_tail(self):
        self.assertEqual(reference_frame_count(288),294)
        self.assertEqual(reference_frame_count(294),294)
        for n in range(5,500):
            rounded=reference_frame_count(n)
            self.assertGreaterEqual(rounded,n);self.assertEqual((rounded-5)%17,0)
            self.assertLess(rounded-n,17)

    def test_v8_mapping_monotone_and_end_preserving(self):
        anchors=read_json(ROOT/'recipes/whiteboard_v35/retime_v8.json')['anchors']
        result=pchip_map(anchors,288)
        self.assertEqual(result['source_frames'][0],0);self.assertEqual(result['source_frames'][-1],287)
        self.assertTrue(all(a<=b for a,b in zip(result['source_frames'],result['source_frames'][1:])))
        self.assertEqual(result['repeat_transitions'],37)

    def test_pchip_matches_scipy(self):
        import numpy as np
        from scipy.interpolate import PchipInterpolator
        anchors=[[0,0],[18,13],[57,39],[83,69],[119,119]]
        result=pchip_map(anchors,120,min_speed=0,max_speed=10)
        expected=PchipInterpolator(*np.asarray(anchors).T)(np.arange(120))
        np.testing.assert_allclose(result['positions'],expected,atol=1e-12)

    def test_reverse_missing_and_extreme_mapping_rejected(self):
        for anchors in [[[0,0],[40,70],[60,50],[119,119]],[[0,0],[1,80],[119,119]]]:
            with self.assertRaises(ValueError):pchip_map(anchors,120)
        events=[{'id':'lift','time_s':2,'confidence':.9,'evidence':'frames'}]
        with self.assertRaisesRegex(ValueError,'Missing'):event_anchors(events,[],120,24)

    def test_simultaneous_identical_correspondences_merge(self):
        sim=[{'id':i,'time_s':2,'confidence':.9,'evidence':'frames'} for i in ['a','b']]
        dit=[dict(e,time_s=1.8) for e in sim]
        self.assertEqual(len(event_anchors(sim,dit,120,24)),3)


class GPUTests(unittest.TestCase):
    def rows(self):
        return [{'index':i,'uuid':f'GPU-{i}','free_mib':60000,'utilization':0,'numa':i//4} for i in range(8)]

    def test_no_cross_numa_fallback(self):
        rows=self.rows()
        for i in [0,1,2,4,5,6]:rows[i]['free_mib']=100
        with self.assertRaisesRegex(RuntimeError,'same-NUMA'):select(rows,2)

    def test_eligible_same_numa_selected_by_uuid(self):
        selected=select(self.rows(),2,allowed=['GPU-5','GPU-6'])
        self.assertEqual([r['index'] for r in selected],[5,6])

    def test_unknown_numa_is_not_assumed_safe(self):
        rows=self.rows()
        for row in rows:row['numa']=-1
        with self.assertRaises(RuntimeError):select(rows,1)

    def test_busy_gpu_excluded(self):
        rows=self.rows();rows[4]['utilization']=95
        with self.assertRaises(RuntimeError):select(rows,4,allowed=['4','5','6','7'])

    def test_lease_is_exclusive_and_released(self):
        with tempfile.TemporaryDirectory() as tmp:
            with Lease(self.rows()[:1],tmp):
                with self.assertRaises(RuntimeError):Lease(self.rows()[:1],tmp)
            with Lease(self.rows()[:1],tmp):pass


class ResidentTests(unittest.TestCase):
    def test_two_different_requests_load_once(self):
        loads=[];requests=[];closed=[]
        class Fake:
            def generate(self,sampling_params_kwargs):requests.append(sampling_params_kwargs);return len(requests)
            def shutdown(self):closed.append(True)
        def factory(**kwargs):loads.append(kwargs);return Fake()
        worker=ResidentGenerator(factory,{'model_path':'not-a-real-model'})
        a,ta=worker.generate({'seed':1,'scene':'kitchen'})
        b,tb=worker.generate({'seed':2,'scene':'workbench'})
        worker.close()
        self.assertEqual((a,b),(1,2));self.assertEqual(len(loads),1)
        self.assertEqual(tb['loading_seconds'],0);self.assertEqual(len(closed),1)

    def test_payload_uses_verified_runtime_schema(self):
        with patch('automation.media.probe',return_value={'width':1024,'height':768,'fps':24,'frames':288,'duration_s':12}):
            value=payload_for({'sim':'/sim','prompt':'scene','appearance':'/image','reference':'/ref','seed':3,'output':'/new'})
        self.assertEqual(value['target'],{'short_edge':768,'aspect_ratio':'4:3','duration_seconds':12})
        self.assertEqual((value['flow_shift'],value['audio_flow_shift']),(12,3))
        self.assertEqual(value['task'],'ref2va')


class GeometryTests(unittest.TestCase):
    def inputs(self):
        import numpy as np
        n=60;t=np.linspace(0,1,n)
        wrists={s:np.stack([np.full(n,x)+.02*np.sin(t),np.full(n,.38),np.full(n,-.18)],axis=1)
                for s,x in [('left',-.15),('right',.15)]}
        backwards={s:np.tile([0,-1.,0],(n,1)) for s in wrists}
        return wrists,backwards

    def test_camera_changes_do_not_move_body(self):
        import numpy as np
        from automation.geometry import build_rig,audit_rig
        wrists,axes=self.inputs();a=scene();b=copy.deepcopy(a)
        b['render_camera']['eye_m']=[-3,2,1]
        ra,_=build_rig(wrists,axes,a,30);rb,_=build_rig(wrists,axes,b,30)
        for key in ra:np.testing.assert_array_equal(ra[key],rb[key])
        self.assertTrue(audit_rig(ra,a,60)['hard_gates_passed'])

    def test_four_operator_sides_and_scene_translations(self):
        import numpy as np
        from scipy.spatial.transform import Rotation
        from automation.geometry import build_rig,audit_rig
        wrists,axes=self.inputs()
        for angle in [0,np.pi/2,np.pi,-np.pi/2]:
            rotation=Rotation.from_euler('z',angle).as_matrix();offset=np.array([2.,-3.,.8])
            data=scene();data['actor']['heading_rad']=angle;data['actor']['root_m']=offset.tolist()
            rig,report=build_rig({s:v@rotation.T+offset for s,v in wrists.items()},
                                 {s:v@rotation.T for s,v in axes.items()},data,30)
            self.assertTrue(audit_rig(rig,data,60)['hard_gates_passed'])
            for side in wrists:
                self.assertLess(report[side]['max_upper_length_error_m'],1e-10)
                self.assertLess(report[side]['max_forearm_length_error_m'],1e-10)

    def test_unreachable_does_not_stretch_or_change_input(self):
        import numpy as np
        from automation.geometry import build_rig
        wrists,axes=self.inputs();wrists['left'][30,0]=2
        original=wrists['left'].copy()
        with self.assertRaisesRegex(ValueError,'unreachable_wrist'):build_rig(wrists,axes,scene(),30)
        np.testing.assert_array_equal(original,wrists['left'])

    def test_report_cannot_hide_wrong_lengths(self):
        from automation.geometry import build_rig,audit_rig
        wrists,axes=self.inputs();rig,_=build_rig(wrists,axes,scene(),30)
        rig['left_elbow'][12,0]+=.1
        with self.assertRaisesRegex(ValueError,'Limb length'):audit_rig(rig,scene(),60)

    def test_arbitrary_object_names_and_short_grips(self):
        import numpy as np
        from automation.geometry import stabilize
        poses={'tool-42':np.tile(np.eye(4),(30,1,1))};poses['tool-42'][:,0,3]=np.linspace(0,.1,30)
        hands={'right':np.tile(np.array([[0,.1,0],[.01,.1,0],[0,.11,0]]),(30,1,1))}
        hands['right']+=poses['tool-42'][:,None,:3,3]
        contacts=[{'object_id':'tool-42','side':'right','start_frame':4,'end_frame':6,'confidence':.9,'evidence':'observed'}]
        filtered,_,report=stabilize(hands,poses,contacts,30)
        self.assertTrue(np.isfinite(filtered['right']).all());self.assertEqual(len(report['contacts']),1)

    def test_overlapping_grasps_rejected(self):
        import numpy as np
        from automation.geometry import stabilize
        poses={'a':np.tile(np.eye(4),(30,1,1)),'b':np.tile(np.eye(4),(30,1,1))}
        hands={'left':np.ones((30,3,3))*.1}
        contacts=[{'object_id':key,'side':'left','start_frame':2,'end_frame':20,'confidence':.9,'evidence':'observed'} for key in poses]
        with self.assertRaisesRegex(ValueError,'Overlapping'):stabilize(hands,poses,contacts,30)

    def test_no_invented_final_release(self):
        import numpy as np
        from automation.event_detection import propose_events
        pose=np.tile(np.eye(4),(120,1,1));pose[40:80,0,3]=np.linspace(0,.2,40);pose[80:,0,3]=.2
        events=propose_events({'renamed-object':pose},[{'object_id':'renamed-object','start_frame':20,'end_frame':119,'confidence':.9}],24)
        self.assertTrue(any('carry' in e['action'] for e in events))
        self.assertFalse(any(e['action']=='release' for e in events))


class ImportTests(unittest.TestCase):
    def test_factory_import_has_no_heavy_dependencies(self):
        command=[sys.executable,'-c',f'import sys;sys.path.insert(0,{str(ROOT)!r});import automation.factory;assert "torch" not in sys.modules;assert "numpy" not in sys.modules']
        subprocess.run(command,check=True)


if __name__=='__main__':unittest.main()
