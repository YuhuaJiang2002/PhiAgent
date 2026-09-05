"""Fixed synthetic lab dressing; immutable MANO/FP motion and foreground masks."""
from runtime import require_launcher
require_launcher()

import sys,json,subprocess,time,hashlib,argparse
from pathlib import Path
import numpy as np
root=Path(__import__('os').environ['PHI_EGO_ROOT']);run=Path(__import__('os').environ['PHI_EGO_RUN'])
sys.path[:0]=[str(root/'envs/sam3d-container-cu128/phiagent-native'),str(root/'third_party/FoundationPose'),str(root/'PhiAgent-PhysicalDemo')]
import cv2,torch,trimesh,nvdiffrast.torch as dr
from PIL import Image
from Utils import make_mesh_tensors,nvdiffrast_render
from render_ego_thirdview import look_at
p=argparse.ArgumentParser();p.add_argument('--clock-upright',action='store_true');p.add_argument('--no-monitor',action='store_true');p.add_argument('--whiteboards',action='store_true');args=p.parse_args()
out=Path(__import__('os').environ['PHI_EGO_OUTPUT'])/'render';out.mkdir(parents=True,exist_ok=False)
state=np.load(run/'fixed-third-view-v6-persistence/reconstruction/world_state_tracks.npz');mano=np.load(run/'reconstruction/hawor_mano_camera_meshes.npz')
R,t=look_at(np.array([.72,-.66,.52]),np.array([.02,-.01,.07]));C=np.eye(4);C[:3,:3]=R;C[:3,3]=t
K=np.array([[720.,0,480],[0,720.,360],[0,0,1.]])
ctx=dr.RasterizeCudaContext();W,H=960,720;N=360
def render(mt,T,light=True):
    with torch.no_grad():
        c,d,_=nvdiffrast_render(K=K,H=H,W=W,ob_in_cams=torch.as_tensor(T[None],device='cuda',dtype=torch.float32),mesh_tensors=mt,glctx=ctx,use_light=light,light_dir=np.array([.3,.5,1.]),w_ambient=.80,w_diffuse=.20)
    return (c[0].cpu().numpy()*255).clip(0,255).astype(np.uint8),d[0].cpu().numpy()
def box(size,center,color):
    m=trimesh.creation.box(extents=size);m.apply_translation(center);m.visual.vertex_colors=np.tile([*color,255],(len(m.vertices),1));return m
def plane(x0,x1,y0,y1,z,image):
    m=trimesh.Trimesh(vertices=[[x0,y0,z],[x1,y0,z],[x1,y1,z],[x0,y1,z]],faces=[[0,1,2],[0,2,3]],process=False)
    m.visual=trimesh.visual.TextureVisuals(uv=[[0,1],[1,1],[1,0],[0,0]],image=Image.fromarray(image));return m
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
foreground_static.append(box([1.04,.72,.045],[0,0,-.037],[229,230,226]))
for x in [-.46,.46]:
    for y in [-.30,.30]:foreground_static.append(box([.045,.045,.71],[x,y,-.413],[111,123,132]))
for x in [-.46,.46]:foreground_static.append(box([.035,.62,.035],[x,0,-.58],[119,132,140]))
foreground_static.append(plane(-.52,.52,-.36,.36,-.012,np.full((512,768,3),[225,229,227],np.uint8)))
aligned=np.load(run/'vggt-omega-512-allframes-v1/table-aligned/vggt_omega_table_aligned.npz')
corners=aligned['mat_corners_source_order_table_world']
xy=np.asarray(corners).reshape(-1,3)[:,:2];lo=xy.min(0);hi=xy.max(0)
foreground_static.append(plane(float(lo[0]),float(hi[0]),float(lo[1]),float(hi[1]),-.010,mattex))
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
def writer(name,width,gray=False):
    return subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','gray' if gray else 'rgb24','-s',f'{width}x{H}','-r','30','-i','-','-an','-c:v','ffv1' if gray else 'libx264',*([] if gray else ['-crf','17','-pix_fmt','yuv420p','-movflags','+faststart']),str(out/name)],stdin=subprocess.PIPE)
vid=writer('fixed_third_view.mp4',W);pairwriter=writer('ego_vs_lab.mp4',2*W);maskwriter=writer('foreground_protection.mkv',W,True)
cap=cv2.VideoCapture(str(run/'input/ego_action_4s_16s.mp4'));start=time.monotonic()
for i in range(N):
    rgb=static_rgb.copy();z=static_z.copy();fg=static_fg.copy();dynamic=[]
    for side in ['left','right']:
        verts=state[f'{side}_mano_vertices_table_world'][i]
        m=trimesh.Trimesh(vertices=verts,faces=mano[f'{side}_faces'],process=False);m.visual.vertex_colors=np.tile([209,159,128,255],(len(verts),1));dynamic.append((verts,make_mesh_tensors(m),C))
    for key,(mesh,mt,data) in objects.items():
        T=data['world_from_object'][i]
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
