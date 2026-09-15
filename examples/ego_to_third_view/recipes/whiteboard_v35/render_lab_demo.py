"""Fixed lab dressing with raw motion or opt-in derived contact stabilization."""
import sys,json,subprocess,time,hashlib,argparse,os
from pathlib import Path
import numpy as np
if not os.getenv('PHI_EGO_OUTPUT'):
    raise SystemExit('Use pipeline.py render_whiteboard_v35; this is a fixed-scene regression recipe, not the generic factory.')
root=Path(os.environ['PHI_EGO_ROOT']);run=Path(os.environ['PHI_EGO_RUN'])
sys.path[:0]=[str(root/'envs/sam3d-container-cu128/phiagent-native'),str(root/'third_party/FoundationPose'),str(root/'PhiAgent-PhysicalDemo')]
import cv2,torch,trimesh,nvdiffrast.torch as dr
from PIL import Image
from Utils import make_mesh_tensors,nvdiffrast_render
from render_ego_thirdview import look_at
p=argparse.ArgumentParser();p.add_argument('--clock-upright',action='store_true');p.add_argument('--no-monitor',action='store_true');p.add_argument('--whiteboards',action='store_true');p.add_argument('--forearms',action='store_true');p.add_argument('--third-person',action='store_true');p.add_argument('--long-edge-arms',action='store_true');p.add_argument('--stabilized-contact',action='store_true');p.add_argument('--fixed-torso',action='store_true');args=p.parse_args()
out=Path(os.environ['PHI_EGO_OUTPUT'])/'render';out.mkdir(parents=True,exist_ok=False)
state=np.load(run/'fixed-third-view-v6-persistence/reconstruction/world_state_tracks.npz');mano=np.load(run/'reconstruction/hawor_mano_camera_meshes.npz')
camera_position=np.array([1.05,-1.40,1.05]) if args.third_person else np.array([.72,-.66,.52])
camera_target=np.array([0.,.08,.24]) if args.third_person else np.array([.02,-.01,.07])
R,t=look_at(camera_position,camera_target);C=np.eye(4);C[:3,:3]=R;C[:3,3]=t
K=np.array([[720.,0,480],[0,720.,360],[0,0,1.]])
ctx=dr.RasterizeCudaContext();W,H=960,720;N=360
def render(mt,T,light=True):
    with torch.no_grad():
        c,d,_=nvdiffrast_render(K=K,H=H,W=W,ob_in_cams=torch.as_tensor(T[None],device='cuda',dtype=torch.float32),mesh_tensors=mt,glctx=ctx,use_light=light,light_dir=np.array([.3,.5,1.]),w_ambient=.80,w_diffuse=.20)
    return (c[0].cpu().numpy()*255).clip(0,255).astype(np.uint8),d[0].cpu().numpy()
def box(size,center,color):
    m=trimesh.creation.box(extents=size);m.apply_translation(center);m.visual.vertex_colors=np.tile([*color,255],(len(m.vertices),1));return m
def ellipsoid(radii,center,color,subdivisions=3):
    m=trimesh.creation.icosphere(subdivisions=subdivisions,radius=1.0);m.apply_scale(radii);m.apply_translation(center);m.visual.vertex_colors=np.tile([*color,255],(len(m.vertices),1));return m
def plane(x0,x1,y0,y1,z,image):
    m=trimesh.Trimesh(vertices=[[x0,y0,z],[x1,y0,z],[x1,y1,z],[x0,y1,z]],faces=[[0,1,2],[0,2,3]],process=False)
    m.visual=trimesh.visual.TextureVisuals(uv=[[0,1],[1,1],[1,0],[0,0]],image=Image.fromarray(image));return m
def tapered_segment(a,b,r0=.030,r1=.060,color=(209,159,128),sections=24):
    """Simple wrist-to-frame forearm control surface; not anatomy evidence."""
    a=np.asarray(a,float);b=np.asarray(b,float);d=b-a;length=np.linalg.norm(d);d/=max(length,1e-9)
    ref=np.array([0.,0.,1.]) if abs(d[2])<.9 else np.array([1.,0.,0.])
    u=np.cross(d,ref);u/=max(np.linalg.norm(u),1e-9);v=np.cross(d,u)
    theta=np.linspace(0,2*np.pi,sections,endpoint=False)
    rings=[]
    for c,radius in [(a,r0),(b,r1)]:rings.append(c+radius*(np.cos(theta)[:,None]*u+np.sin(theta)[:,None]*v))
    verts=np.concatenate(rings);faces=[]
    for j in range(sections):
        k=(j+1)%sections;faces.extend([[j,k,sections+k],[j,sections+k,sections+j]])
    ca=len(verts);cb=ca+1;verts=np.vstack([verts,a,b])
    for j in range(sections):
        k=(j+1)%sections;faces.extend([[ca,k,j],[cb,sections+j,sections+k]])
    m=trimesh.Trimesh(vertices=verts,faces=faces,process=False);m.visual.vertex_colors=np.tile([*color,255],(len(verts),1));return m
# Estimate material colour from observed green mat pixels, discard hands/book.
observed=np.array(Image.open(run/'observed-table-texture-v2/table_texture.png').convert('RGB'))
hsv=cv2.cvtColor(observed,cv2.COLOR_RGB2HSV);good=(hsv[...,0]>25)&(hsv[...,0]<55)&(hsv[...,1]>30)&(hsv[...,1]<130)&(hsv[...,2]>65)&(hsv[...,2]<180)
mat_color=np.median(observed[good],axis=0) if good.sum()>100 else np.array([107,118,91])
rng=np.random.default_rng(20260905);noise=cv2.GaussianBlur(rng.normal(0,1,(450,700)).astype(np.float32),(0,0),.7)
mattex=np.clip(mat_color[None,None,:]+noise[...,None]*1.1,0,255).astype(np.uint8)
Image.fromarray(mattex).save(out/'clean_mat_material.png')
background=[]
background.append(box([8,8,.05],[0,0,-.795],[194,198,202]))
background.append(box([5,.08,3.0],[0,1.28,.67],[222,228,231]))
background.append(box([.08,5,3.0],[-1.45,0,.67],[218,224,228]))
# Rear workbench and cabinets, spatially separate from interaction table.
background.append(box([2.35,.44,.045],[-.10,.92,-.10],[218,225,231]))
for x in [-.96,-.50,-.04,.42,.88]:
    background.append(box([.43,.39,.63],[x,.93,-.437],[185,198,207]))
    background.append(box([.25,.018,.018],[x,.718,-.18],[84,100,111]))
    background.append(box([.25,.018,.018],[x,.718,-.42],[84,100,111]))
# Wall-mounted whiteboard, frame, shelf, modest lab apparatus.
background.append(box([1.05,.025,.51],[-.16,1.22,.66],[119,140,153]))
background.append(box([1.00,.028,.46],[-.16,1.202,.66],[236,241,240]))
background.append(box([.72,.17,.024],[.76,1.14,.43],[167,182,192]))
for j in range(4):background.append(box([.075,.09,.14+.02*j],[.52+.12*j,1.12,.51+.01*j],[144+12*j,166+8*j,181+5*j]))
monitor=[box([.32,.07,.23],[-.78,.93,.075],[65,80,91]),box([.275,.075,.185],[-.78,.884,.08],[119,157,173]),box([.035,.035,.10],[-.78,.94,-.035],[98,113,122])]
if not args.no_monitor:background.extend(monitor)
else:
    removal=np.zeros((H,W),np.uint8)
    for m in monitor:
        _,depth=render(make_mesh_tensors(m),C);removal[depth>0]=255
    cv2.imwrite(str(out/'monitor_removal_mask.png'),removal)
foreground_static=[]
if args.whiteboards:
    background=[box([8,8,.05],[0,0,-.795],[205,208,210]),box([5,.05,3],[0,1.35,.67],[225,229,230])]
    # Two tall plain white partitions, matching the visible source scene type.
    for x in [-.44,.44]:
        background.append(box([.87,.04,1.95],[x,.73,.215],[186,191,192]))
        background.append(box([.842,.044,1.922],[x,.728,.215],[239,240,238]))
if args.third_person:
    # Move the partitions behind the actor.  The actor is deliberately visible
    # from an external camera, so the hands cannot read as camera-wearer hands.
    background=[box([8,8,.05],[0,0,-.795],[205,208,210]),box([5,.05,3],[0,1.48,.67],[225,229,230])]
    for x in [-.52,.52]:
        background.append(box([1.02,.04,1.95],[x,1.43,.215],[186,191,192]))
        background.append(box([.992,.044,1.922],[x,1.428,.215],[239,240,238]))
foreground_static.append(box([1.04,.72,.045],[0,0,-.037],[229,230,226]))
for x in [-.46,.46]:
    for y in [-.30,.30]:foreground_static.append(box([.045,.045,.71],[x,y,-.413],[111,123,132]))
for x in [-.46,.46]:foreground_static.append(box([.035,.62,.035],[x,0,-.58],[119,132,140]))
foreground_static.append(plane(-.52,.52,-.36,.36,-.012,np.full((512,768,3),[225,229,227],np.uint8)))
aligned=np.load(run/'vggt-omega-512-allframes-v1/table-aligned/vggt_omega_table_aligned.npz')
corners=aligned['mat_corners_source_order_table_world']
xy=np.asarray(corners).reshape(-1,3)[:,:2];lo=xy.min(0);hi=xy.max(0)
foreground_static.append(plane(float(lo[0]),float(hi[0]),float(lo[1]),float(hi[1]),-.010,mattex))
if args.third_person:
    # A complete, visibly connected upper body on the far side of the table.
    # The table occludes the lower torso naturally from this external view.
    foreground_static.extend([
        ellipsoid([.205,.115,.245],[.02,.405,.305],[69,92,118]),
        tapered_segment(np.array([.02,.385,.515]),np.array([.02,.385,.565]),.045,.055,(209,159,128)),
        ellipsoid([.125,.105,.145],[.02,.375,.685],[209,159,128]),
        ellipsoid([.128,.108,.085],[.02,.400,.765],[54,43,37]),
    ])
static_rgb=np.full((H,W,3),235,np.uint8);static_z=np.full((H,W),np.inf,np.float32);static_fg=np.zeros((H,W),np.uint8)
for is_fg,meshes in [(False,background),(True,foreground_static)]:
    for m in meshes:
        c,d=render(make_mesh_tensors(m),C);sel=(d>0)&(d<static_z);static_rgb[sel]=c[sel];static_z[sel]=d[sel];static_fg[sel]=255 if is_fg else 0
cv2.imwrite(str(out/'static_lab.jpg'),static_rgb[...,::-1])
cv2.imwrite(str(out/'static_lab.png'),static_rgb[...,::-1])
objects={};hashes={}
for key in ['alarm_clock','small_cylinder','tall_cylinder']:
    folder=run/'foundationpose-layout-v2'/key;mesh=trimesh.load(folder/'tracking_mesh_metric.obj',force='mesh')
    path=run/('foundationpose-clock-upright-v4' if args.clock_upright else 'foundationpose-multiview-persistent-v3')/key/'foundationpose_tracks.npz'
    objects[key]=(mesh,make_mesh_tensors(mesh),np.load(path));hashes[key]=hashlib.sha256(path.read_bytes()).hexdigest()
stable_hands,stable_poses=None,None
if args.fixed_torso:
    tracks=np.load(run/'fixed-third-view-v34-stabilized-contact/render/stabilized_tracks.npz')
    stable_hands={s:tracks[s+'_hands'] for s in ['left','right']}
    stable_poses={key:tracks[key] for key in objects}
    from fixed_torso_rig import build_rig
    body_rig=build_rig(stable_hands,mano,out)
    # One rigid elliptical trunk, shared by both shoulders. No duplicate shirts.
    theta=np.linspace(0,2*np.pi,32,endpoint=False)
    torso_vertices=np.concatenate([np.stack([rx*np.cos(theta),ry*np.sin(theta),np.full(32,z)],axis=1) for z,rx,ry in [(-.48,.17,.105),(-.20,.21,.13),(.015,.205,.12)]])
    torso_faces=[]
    for ring in range(2):
        for j in range(32):
            k=(j+1)%32;a=ring*32+j;b=ring*32+k;c=(ring+1)*32+k;d=(ring+1)*32+j
            torso_faces.extend([[a,b,c],[a,c,d]])
    for j in range(1,31):torso_faces.extend([[0,j+1,j],[64,64+j,64+j+1]])
elif args.stabilized_contact:
    from stabilize_contact import stabilize
    stable_hands,stable_poses=stabilize(state,objects,out)
def writer(name,width,gray=False):
    return subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','gray' if gray else 'rgb24','-s',f'{width}x{H}','-r','30','-i','-','-an','-c:v','ffv1' if gray else 'libx264',*([] if gray else ['-crf','17','-pix_fmt','yuv420p','-movflags','+faststart']),str(out/name)],stdin=subprocess.PIPE)
vid=writer('fixed_third_view.mp4',W);pairwriter=writer('ego_vs_lab.mp4',2*W);maskwriter=writer('foreground_protection.mkv',W,True)
cap=cv2.VideoCapture(str(run/'input/ego_action_4s_16s.mp4'));start=time.monotonic()
for i in range(N):
    rgb=static_rgb.copy();z=static_z.copy();fg=static_fg.copy();dynamic=[]
    if args.fixed_torso:
        torso=trimesh.Trimesh(vertices=torso_vertices@body_rig['rotation'][i].T+body_rig['root'],faces=torso_faces,process=False)
        torso.visual.vertex_colors=np.tile([69,92,118,255],(len(torso.vertices),1))
        dynamic.append((torso.vertices,make_mesh_tensors(torso),C))
    hands=[]
    for side in ['left','right']:
        verts=stable_hands[side][i] if stable_hands is not None else state[f'{side}_mano_vertices_table_world'][i]
        faces=mano[f'{side}_faces'];m=trimesh.Trimesh(vertices=verts,faces=faces,process=False);m.visual.vertex_colors=np.tile([209,159,128,255],(len(verts),1));hands.append((verts,make_mesh_tensors(m),C))
        if args.forearms or args.third_person or args.long_edge_arms:
            edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
            unique,count=np.unique(edges,axis=0,return_counts=True);boundary=np.unique(unique[count==1])
            wrist=verts[boundary].mean(0)
        if args.fixed_torso:
            shoulder=body_rig[side+'_shoulder'][i];elbow=body_rig[side+'_elbow'][i];wrist=body_rig[side+'_wrist'][i]
            sleeve_end=shoulder+.42*(elbow-shoulder)
            for limb in [tapered_segment(shoulder,sleeve_end,.068,.052,(69,92,118)),tapered_segment(sleeve_end,elbow,.046,.037),ellipsoid([.037,.037,.037],elbow,[209,159,128]),tapered_segment(elbow,wrist,.037,.023)]:
                dynamic.append((limb.vertices,make_mesh_tensors(limb),C))
        elif args.long_edge_arms:
            backwards=wrist-verts.mean(0);backwards/=np.linalg.norm(backwards)
            elbow=wrist+.28*backwards
            proximal=np.array([.12 if side=='left' else -.12,-1.,.28]);proximal/=np.linalg.norm(proximal)
            shoulder=elbow+.31*proximal
            sleeve_end=shoulder+.48*(elbow-shoulder)
            # Only the cropped shirt side is visible: join each sleeve into
            # fabric outside the same physical long edge, not into the camera.
            shirt=box([.15,.60,.18],shoulder+np.array([0.,-.27,0.]),[69,92,118])
            for limb in [shirt,tapered_segment(shoulder,sleeve_end,.065,.052,(69,92,118)),tapered_segment(sleeve_end,elbow,.047,.037),ellipsoid([.037,.037,.037],elbow,[209,159,128]),tapered_segment(elbow,wrist,.037,.023)]:
                dynamic.append((limb.vertices,make_mesh_tensors(limb),C))
        if args.forearms and not args.third_person:
            cam=wrist@R.T+t;centroid=verts.mean(0)@R.T+t
            wrist_uv=np.array([K[0,0]*cam[0]/cam[2]+K[0,2],K[1,1]*cam[1]/cam[2]+K[1,2]])
            center_uv=np.array([K[0,0]*centroid[0]/centroid[2]+K[0,2],K[1,1]*centroid[1]/centroid[2]+K[1,2]])
            outdir=wrist_uv-center_uv;outdir/=max(np.linalg.norm(outdir),1e-9)
            # The operator stands at the near side of the table. Route both
            # forearms to nearby, separated points just below the bottom edge;
            # this keeps projected limb length human-scale instead of stretching
            # wrists all the way across the frame to a distant side boundary.
            target_v=800.
            target_u=np.clip(wrist_uv[0]+(-95. if side=='left' else 95.),80.,880.)
            ray=np.array([(target_u-K[0,2])/K[0,0],(target_v-K[1,2])/K[1,1],1.])
            target_z=.065
            denominator=np.dot(ray,R[:,2])
            if abs(denominator)<1e-6:raise RuntimeError('arm endpoint ray is parallel to tabletop')
            ray_scale=(target_z+np.dot(t,R[:,2]))/denominator
            if ray_scale<=0:raise RuntimeError('arm endpoint lies behind the camera')
            target_cam=ray*ray_scale
            target=(target_cam-t)@R
            target=wrist+(target-wrist)*1.05;direction=target-wrist
            arm=tapered_segment(wrist-direction/np.linalg.norm(direction)*.018,target)
            dynamic.append((arm.vertices,make_mesh_tensors(arm),C))
        if args.third_person:
            # True third-person kinematic chain: each wrist connects through an
            # elbow to a fixed shoulder on the visible torso across the table.
            shoulder=np.array([-.165 if side=='left' else .205,.34,.505])
            lateral=-.055 if side=='left' else .055
            elbow=.50*shoulder+.50*wrist+np.array([lateral,-.010,.060])
            upper=tapered_segment(shoulder,elbow,.052,.038,(209,159,128))
            lower=tapered_segment(elbow,wrist,.038,.026,(209,159,128))
            dynamic.extend([(upper.vertices,make_mesh_tensors(upper),C),(lower.vertices,make_mesh_tensors(lower),C)])
    dynamic.extend(hands)
    for key,(mesh,mt,data) in objects.items():
        T=stable_poses[key][i] if stable_poses is not None else data['world_from_object'][i]
        if not data['valid'][i]:raise RuntimeError(f'Invalid pose {key} {i}')
        verts=mesh.vertices@T[:3,:3].T+T[:3,3];dynamic.append((verts,mt,C@T))
    # Soft projected shadows, confined to the known tabletop surface.
    shadow=np.zeros((H,W),np.float32)
    for verts,_,_ in dynamic:
        pts=verts.copy();height=np.maximum(pts[:,2]+.009,0);pts[:,0]-=.25*height;pts[:,1]-=.35*height;pts[:,2]=-.009
        cam=pts@R.T+t;uv=cam@K.T;uv=uv[:,:2]/uv[:,2:]
        hull=cv2.convexHull(uv.astype(np.float32)).astype(np.int32)
        layer=np.zeros((H,W),np.float32);cv2.fillConvexPoly(layer,hull,.20)
        layer=cv2.GaussianBlur(layer,(0,0),max(2.,float(np.median(height))*30))
        shadow=np.maximum(shadow,layer)
    yy,xx=np.mgrid[:H,:W];rays=np.stack([(xx-480)/720,(yy-360)/720,np.ones((H,W))],-1)
    world=(rays*static_z[...,None]-t)@R
    tabletop=(static_fg>0)&np.isfinite(static_z)&(np.abs(world[...,2]+.011)<.01)
    rgb[tabletop]=(rgb[tabletop]*(1-shadow[tabletop,None])).astype(np.uint8)
    for _,mt,T in dynamic:
        c,d=render(mt,T);sel=(d>0)&(d<z);rgb[sel]=c[sel];z[sel]=d[sel];fg[sel]=255
    ok,source=cap.read()
    if not ok:raise RuntimeError('Source frame missing')
    source=cv2.cvtColor(cv2.resize(source,(W,H)),cv2.COLOR_BGR2RGB);pair=np.concatenate([source,rgb],axis=1)
    cv2.putText(pair,'Original ego video',(18,30),0,.65,(255,255,255),2);cv2.putText(pair,'Reconstructed motion | synthetic lab',(W+18,30),0,.58,(35,45,55),1)
    vid.stdin.write(rgb.tobytes());pairwriter.stdin.write(pair.tobytes());maskwriter.stdin.write(fg.tobytes())
    if i in [0,120,240,330]:cv2.imwrite(str(out/f'comparison_{i:04d}.jpg'),pair[...,::-1])
    if i%60==0:print('frame',i,'elapsed',time.monotonic()-start,flush=True)
cap.release()
for proc in [vid,pairwriter,maskwriter]:
    proc.stdin.close()
    if proc.wait()!=0:raise RuntimeError('Video encoding failed')
(out/'manifest.json').write_text(json.dumps({'frames':N,'fps':30,'duration_s':12,'motion_source_sha256':hashes,'mat_color_from_observation_rgb':mat_color.tolist(),'scene':'Synthetic fixed lab, not reconstruction of unseen room. Mat material colour estimated from observations; book/occluder residuals intentionally removed.','shadows':'Approximate soft projected shadows, not contact/physics evidence.','foreground_mask':'Table including contact region, hands and all objects; lossless FFV1.','elapsed_s':time.monotonic()-start},indent=2))
print('COMPLETE',out,flush=True)
