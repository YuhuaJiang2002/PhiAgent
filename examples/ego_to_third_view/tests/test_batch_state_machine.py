"""CPU state-machine tests with explicit fake media; these are not video quality tests."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from automation.contracts import digest,read_json,write_json
from automation.factory import Factory,file_bindings
from test_automation import scene,review

META={'frames':120,'fps':24.,'width':512,'height':384,'duration_s':5.}


def packet(videos,folder,events):
    folder.mkdir();write_json(folder/'packet.json',{'test_fake':True})
    return {'test_fake':True}


class FakeStages(Factory):
    """Real numeric rig/contracts and filesystem state; intentionally fake video bytes."""
    def run_adapter(self,stage,clip,folder,state):
        import numpy as np
        from automation.geometry import build_rig
        target=self.new_stage(folder,state,stage)
        if stage=='reconstruct':
            value=scene();value['source_sha256']=state['source_sha256']
            (target/'mug.obj').write_text('# fake mesh for state-machine tests\n')
            value['objects'][0]['mesh']=str(target/'mug.obj')
            write_json(target/'scene.json',value)
            events={'frames':120,'fps':24,'source_sha256':state['source_sha256'],
                    'events':[{'id':'transfer','time_s':2.5,'confidence':.9,'evidence':'test only'}],'contacts':[]}
            write_json(target/'events.json',events)
            wrists={s:np.tile([x,.38,-.18],(120,1)) for s,x in [('left',-.15),('right',.15)]}
            axes={s:np.tile([0,-1.,0],(120,1)) for s in wrists}
            rig,report=build_rig(wrists,axes,value,24)
            np.savez(target/'rig.npz',**rig)
            vertices=np.array([[-.01,0,-.01],[.01,0,-.01],[.01,0,.01],[-.01,0,.01],[0,.08,0]])
            faces=np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
            np.savez(target/'motion.npz',timestamps_s=np.arange(120)/24,
                left_hand_world_m=wrists['left'][:,None]+vertices,right_hand_world_m=wrists['right'][:,None]+vertices,
                left_faces=faces,right_faces=faces,**{'arbitrary-mug__world_from_object':np.tile(np.eye(4),(120,1,1))})
            write_json(target/'geometry_report.json',{'rig':report})
            (target/'ego.mp4').write_bytes(b'fake ego '+clip['id'].encode())
            result={key:str(target/file) for key,file in [('scene','scene.json'),('events','events.json'),
                    ('motion','motion.npz'),('rig','rig.npz'),('geometry_report','geometry_report.json'),('ego','ego.mp4')]}
        else:
            previous='reconstruct' if stage=='stabilize' else 'stabilize'
            result=copy.deepcopy(state['artifacts'][previous])
            if stage=='render':
                (target/'sim.mp4').write_bytes(b'fake sim '+clip['id'].encode());result['sim']=str(target/'sim.mp4')
        self.validate_stage(stage,result,state)
        state['artifacts'][stage]=result;state['hashes'][stage]=file_bindings(result)

    def generate(self,clip,folder,state):
        target=self.new_stage(folder,state,'generate');(target/'dit.mp4').write_bytes(b'fake dit '+clip['id'].encode())
        state['artifacts']['generate']={'dit':str(target/'dit.mp4')}
        state['hashes']['generate']=file_bindings(state['artifacts']['generate'])


class StateMachineTests(unittest.TestCase):
    def test_external_asset_changes_invalidate_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);value=scene()
            (root/'mug.obj').write_text('# mesh\n')
            (root/'surface.png').write_bytes(b'texture-before')
            write_json(root/'scene.json',value)
            artifacts={'scene':str(root/'scene.json')}
            before=file_bindings(artifacts)
            (root/'surface.png').write_bytes(b'texture-after')
            self.assertNotEqual(before,file_bindings(artifacts))

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.source=self.root/'source.mp4';self.source.write_bytes(b'fake input')
        self.manifest={'schema_version':1,'clips':[{'id':'kitchen','source':str(self.source)},{'id':'workbench','source':str(self.source)}]}
        self.batch=self.root/'batch'
        self.patches=[patch('automation.factory.probe',return_value=META),patch('automation.factory.review_packet',side_effect=packet),
            patch('automation.factory.threeway',side_effect=lambda a,b,c,d:Path(d).write_bytes(b'fake threeway')),
            patch('automation.factory.retime',side_effect=lambda a,b,c:Path(b).write_bytes(b'fake retimed'))]
        for p in self.patches:p.start()

    def tearDown(self):
        for p in self.patches:p.stop()
        self.tmp.cleanup()

    def factory(self,resume=True):return FakeStages(self.manifest,self.batch,resume=resume)

    def answer(self,clip,decision='accept',issues=None):
        state=read_json(self.batch/clip/'state.json');pending=state['pending_review']
        value=review(pending['binding'],pending['phase'],decision,issues)
        if decision=='repair':
            value['checks']['timing']='fail'
            value['dit_events']=[{'id':'transfer','time_s':2.,'confidence':.9,'evidence':'test only'}]
        write_json(Path(pending['directory'])/'review.json',value)

    def test_two_clips_wait_before_gpu_and_resume_independently(self):
        result=self.factory(False).run()
        self.assertTrue(all(r['status']=='NEEDS_REVIEW' for r in result.values()))
        self.assertFalse(list(self.batch.rglob('generate-*')))
        self.answer('kitchen')
        result=self.factory().run()
        self.assertEqual(result['kitchen']['step'],'post_review')
        self.assertEqual(result['workbench']['step'],'pre_review')
        self.answer('kitchen');self.answer('workbench')
        result=self.factory().run()
        self.assertEqual(result['kitchen']['status'],'COMPLETE')
        self.assertEqual(result['workbench']['step'],'post_review')
        self.answer('workbench');result=self.factory().run()
        self.assertTrue(all(r['status']=='COMPLETE' for r in result.values()))
        self.assertEqual(len(list(self.batch.glob('kitchen/generate-*'))),1)

    def test_timing_only_repair_does_not_regenerate_sim_or_h3(self):
        self.factory(False).run();self.answer('kitchen');self.factory().run()
        self.answer('kitchen','repair',[{'code':'dit_timing','evidence':'test offset'}])
        result=self.factory().run()
        self.assertEqual(result['kitchen']['status'],'NEEDS_REVIEW')
        state=read_json(self.batch/'kitchen/state.json')
        self.assertEqual(state['attempts']['generate'],1);self.assertEqual(state['attempts']['render'],1)
        self.assertEqual(state['attempts']['align'],1);self.assertEqual(state['repairs'],1)
        self.answer('kitchen');self.assertEqual(self.factory().run()['kitchen']['status'],'COMPLETE')

    def test_geometry_repair_invalidates_all_downstream(self):
        self.factory(False).run();self.answer('kitchen','repair',[{'code':'wrong_actor_origin','evidence':'wrong source side'}])
        result=self.factory().run();state=read_json(self.batch/'kitchen/state.json')
        self.assertEqual(result['kitchen']['status'],'NEEDS_REVIEW')
        self.assertEqual(state['attempts']['reconstruct'],2);self.assertEqual(state['attempts']['render'],2)
        self.assertNotIn('generate',state['artifacts'])

    def test_changed_artifact_cannot_reuse_review(self):
        self.factory(False).run();self.answer('kitchen')
        state=read_json(self.batch/'kitchen/state.json');Path(state['artifacts']['render']['sim']).write_bytes(b'changed')
        result=self.factory().run()
        self.assertEqual(result['kitchen']['status'],'FAILED');self.assertIn('Artifact changed',result['kitchen']['error'])

    def test_review_budget_stops_loop(self):
        self.manifest['max_repairs']=0;self.factory(False).run()
        self.answer('kitchen','repair',[{'code':'contact_slip','evidence':'visible gap'}])
        self.assertEqual(self.factory().run()['kitchen']['status'],'REPAIR_BUDGET_EXHAUSTED')

    def test_missing_one_source_does_not_stop_other_clip(self):
        self.manifest['clips'][0]['source']=str(self.root/'missing.mp4')
        result=self.factory(False).run()
        self.assertEqual(result['kitchen']['status'],'FAILED');self.assertEqual(result['workbench']['status'],'NEEDS_REVIEW')

    def test_changed_manifest_requires_new_batch(self):
        self.factory(False).run();self.manifest['max_repairs']=3
        with self.assertRaisesRegex(ValueError,'Manifest changed'):self.factory()


if __name__=='__main__':unittest.main()
