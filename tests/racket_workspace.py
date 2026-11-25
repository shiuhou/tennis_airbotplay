"""Sample mounted racket workspace using existing FK and collision checks (no motion)."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.stats import qmc
from airbot_tt.config import ROOT
from airbot_tt.collision import stl_triangles
from deepmind_audit import search_context, source_hashes

CONTEXT=None


def check_pose(q):
    model,checker,*_=CONTEXT
    collisions=checker.check(q)
    return collisions


def face_samples(model):
    """Sample only the approved planar face above the existing handle/blade split."""
    g=model.tool_geometry['components']['paddle']
    triangles=stl_triangles(ROOT/g['mesh'])
    t=model.link6_tcp
    local=(triangles-t[:3,3])@t[:3,:3]
    mask=abs(local[:,:,2]).max(axis=1)<2e-5
    clipped=[];cut=g['collision_split_link6_z_m']
    # Clip crossing CAD triangles rather than discard their whole face area.
    for triangle in triangles[mask]:
        polygon=[]
        for a,b in zip(triangle,np.roll(triangle,-1,axis=0)):
            inside_a=a[2]>=cut;inside_b=b[2]>=cut
            if inside_a:polygon.append(a)
            if inside_a!=inside_b:polygon.append(a+(b-a)*((cut-a[2])/(b[2]-a[2])))
        for i in range(1,len(polygon)-1):clipped.append([polygon[0],polygon[i],polygon[i+1]])
    tri=(np.array(clipped)-t[:3,3])@t[:3,:3]
    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    assert len(tri)>0 and area.sum()>.01
    rng=np.random.default_rng(8202)
    chosen=rng.choice(len(tri),256,p=area/area.sum())
    uv=rng.random((256,2));uv[uv.sum(axis=1)>1]=1-uv[uv.sum(axis=1)>1]
    points=tri[chosen,0]+uv[:,:1]*(tri[chosen,1]-tri[chosen,0])+uv[:,1:]*(tri[chosen,2]-tri[chosen,0])
    # Retain original planar face vertices as well as interior samples.
    points=np.unique(np.r_[points,tri.reshape(-1,3)],axis=0)
    assert abs(points[:,2]).max()<2e-5
    return points,float(area.sum())


def run(args):
    import multiprocessing
    global CONTEXT
    start=time.monotonic();CONTEXT=search_context()
    model,checker,bc,strike,rl,shot,base=CONTEXT
    if args.base_position is not None:
        base=np.array(args.base_position);checker.base_position=base
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'workspace.json').exists():raise FileExistsError('Existing workspace evidence is preserved; choose another output for another installation')
    nominal=np.array(strike['ready_q_rad']);scale=np.array(rl['action_scale_rad'])
    if args.action_radius is not None:scale=np.full(6,args.action_radius)
    bounds={'current':(np.maximum(model.lower,nominal-scale),np.minimum(model.upper,nominal+scale)),
            'full':(model.lower,model.upper)}
    if args.only_current:bounds={'current':bounds['current']}
    points,area=face_samples(model)
    protocol=dict(schema='mounted-racket-static-workspace-v1',base_position_m=base.tolist(),base_rotation='identity, unchanged',
        action_scale_rad=scale.tolist(),installation_support='Not modeled; translated base requires independent support at unchanged height',
        sample_power=args.power,seed=8201,poses_per_box=2**args.power,voxel_m=.02,
        face_point_count=len(points),face_sample_area_m2=area,ball_radius_m=bc['radius_m'],
        face='Approved positive TCP-Z planar blade only; original vertices plus 256 area-weighted samples; excludes handle/edge/back face',
        source_sha256=source_hashes(),geometry_sha256=hashlib.sha256((ROOT/'assets/derived/racket/geometry.json').read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Sampled collision-free static postures, not connected paths, timing, ball approach clearance, or legal return guarantees',
        joint_bounds_rad={k:np.array(v).tolist() for k,v in bounds.items()})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    sample=qmc.Sobol(6,scramble=True,seed=8201).random_base2(args.power)
    data={};reports={}
    for name,(lower,upper) in bounds.items():
        qs=np.r_[nominal[None],qmc.scale(sample,lower,upper)]
        collisions=[]
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork')) as pool:
            for i,pairs in enumerate(pool.map(check_pose,qs,chunksize=16),1):
                collisions.append(pairs)
                if i%1024==0:print('WORKSPACE',name,i,len(qs),flush=True)
        ok=np.array([not p for p in collisions]);q=qs[ok]
        # Exact inclusion of current clear samples in full makes set comparisons honest.
        if name=='full':q=np.unique(np.r_[q,data['current_q']],axis=0)
        poses=np.array([model.fk(x) for x in q]);tcp=poses[:,:3,3]+base;rot=poses[:,:3,:3];normals=rot[:,:,2]
        assert np.allclose(np.linalg.norm(normals,axis=1),1)
        contacts=np.einsum('nij,pj->npi',rot,points+[0,0,bc['radius_m']])+tcp[:,None,:]
        # Sphere against world boxes: filter table/net/ground penetration; arm occlusion not checked.
        valid=contacts[:,:,2]>bc['radius_m']
        for box in checker.obstacles:
            lo=np.array(box.center_m)-np.array(box.size_m)/2;hi=np.array(box.center_m)+np.array(box.size_m)/2
            nearest=np.clip(contacts,lo,hi)
            valid &= np.linalg.norm(contacts-nearest,axis=2)>=bc['radius_m']-1e-8
        pose_index,face_index=np.nonzero(valid);xyz=contacts[valid]
        vox=np.floor(xyz/.02).astype(np.int32)
        unique,idx=np.unique(vox,axis=0,return_index=True)
        data.update({name+'_q':q,name+'_tcp':tcp,name+'_normal':normals,name+'_rotation':rot,
                     name+'_contact_voxel':unique,name+'_contact_xyz':xyz[idx],
                     name+'_contact_pose_index':pose_index[idx],name+'_contact_face_index':face_index[idx]})
        # All q, normals and face samples remain available: voxel representatives do not imply unique orientation.
        reports[name]=dict(sampled=len(qs),sampled_clear=int(ok.sum()),sampled_collision=int((~ok).sum()),
            clear_with_current_inclusion=len(q),collision_pairs=dict(Counter('/'.join(p) for ps in collisions for p in ps)),
            tcp_bounds_m=[tcp.min(0).tolist(),tcp.max(0).tolist()],contact_bounds_m=[xyz.min(0).tolist(),xyz.max(0).tolist()],
            contact_voxels=len(unique),forward_normal_pose_count=int((normals[:,0]>0).sum()))
        print('WORKSPACE_RESULT',name,reports[name],flush=True)
    data['face_local_m']=points;data['ready_q']=nominal
    np.savez_compressed(out/'samples.npz',**data)
    assert protocol['source_sha256']==source_hashes()
    report=dict(status='PASS',protocol=protocol,sets=reports,seconds=time.monotonic()-start,
        samples_sha256=hashlib.sha256((out/'samples.npz').read_bytes()).hexdigest(),
        limits=['Sample extrema are not analytic boundaries; empty voxels are not unreachable certificates',
                'Static arm collision only; no path connectivity, time-to-contact or ball-versus-arm occlusion test',
                'Positive face surface samples only, not the full swept solid or both faces; no spin/return inference',
                'Base location is repository configuration, not measured hardware registration'])
    (out/'workspace.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def compare(args):
    """Nine predeclared paired geometry conditions; no changes to production config."""
    from argparse import Namespace
    base=args.output;base.mkdir(parents=True,exist_ok=True)
    destination=base/'comparison.json'
    if destination.exists():raise FileExistsError('Preserve completed comparison; use another output directory')
    protocol=dict(radii_rad=[.3,.6,1.0],retreat_m=[0,.15,.3],base_initial=[-1.275,0,.76],power=args.power,seed=8201,
        scope='Paired static geometry; support at unchanged height assumed but not modeled; no control changes')
    # Derive the reference position from the same repository installation configuration.
    protocol['base_initial']=json.loads((ROOT/'configs/single_shot.json').read_text())['base_position_m']
    (base/'comparison_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    results=[]
    for retreat in protocol['retreat_m']:
        for radius in protocol['radii_rad']:
            initial=np.array(protocol['base_initial']);initial[0]-=retreat
            baseline=base/'workspace.json'
            if retreat==0 and radius==.6 and baseline.exists():
                path=baseline;report=json.loads(path.read_text())
                assert report['protocol']['sample_power']==args.power
                assert report['protocol']['source_sha256']==source_hashes()
                np.testing.assert_allclose(report['protocol']['base_position_m'],initial)
            else:
                output=base/f'r{radius:.1f}_back{retreat:.2f}'
                report=run(Namespace(power=args.power,workers=args.workers,base_position=initial.tolist(),
                    action_radius=radius,only_current=True,output=output))
                path=output/'workspace.json'
            results.append(dict(radius_rad=radius,retreat_m=retreat,path=str(path.resolve()),result=report['sets']['current']))
            print('COMPARISON_FINISHED',retreat,radius,flush=True)
    destination.write_text(json.dumps(dict(status='PASS',protocol=protocol,rows=results),indent=2)+'\n')



if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--power',type=int,default=12)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--base-position',nargs=3,type=float)
    parser.add_argument('--action-radius',type=float,help='Diagnostic symmetric radius for all six joints; does not edit PPO')
    parser.add_argument('--compare',action='store_true',help='Run fixed 3 radii by 3 retreats at configured base height')
    parser.add_argument('--only-current',action='store_true',help='Skip full-URDF reference when comparing action boxes')
    parser.add_argument('--output',type=Path,default=ROOT/'.agents/runs/foundation/workspace')
    args=parser.parse_args()
    if not 8<=args.power<=16:parser.error('power must be 8..16')
    if args.action_radius is not None and (not np.isfinite(args.action_radius) or args.action_radius<=0):parser.error('action radius must be positive and finite')
    if args.compare:
        if args.action_radius is not None or args.base_position is not None:parser.error('--compare uses its own fixed paired parameters')
        compare(args)
    else:run(args)
