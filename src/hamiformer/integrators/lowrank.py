import sys
import torch
from types import MethodType

def install_lowrank(runtime, rank=2, probes=4, chunk=128, cache=True, batch_ad=True, jet_graph=True, joint_probes=False, compiled=True, install_symbols=True):
    from hamiformer.training import hamiballs_formal as training
    from hamiformer.physics.hamiballs_type2 import HamiBallsAffineJets
    model=runtime.models['h'];m=model.state_dim;r=min(rank,2*m)
    if not compiled:
        def validate(obj,q,p,c):
            if q.shape != p.shape or q.shape[-1] != obj.state_dim:
                raise ValueError('Hamiltonian state shape mismatch')
        model._validate=MethodType(validate,model)
    def explicit_norm(obj,x):
        centered=x-x.mean(-1,keepdim=True)
        return centered*torch.rsqrt(centered.square().mean(-1,keepdim=True)+obj.eps)*obj.weight+obj.bias
    for layer in model.modules():
        if isinstance(layer,torch.nn.LayerNorm):layer.forward=MethodType(explicit_norm,layer)
    from hamiformer.inference.derivatives import pointwise_attention
    pointwise_attention(model)
    device=next(model.parameters()).device
    generator=torch.Generator(device=device).manual_seed(42)
    dtype=next(model.parameters()).dtype
    signs=(torch.randint(0,2,(2*m,probes),device=device,generator=generator)*2-1).to(dtype)
    omega=torch.randn(2*m,r,device=device,dtype=dtype,generator=generator)
    runtime.lowrank_directions={'signs':signs.detach().cpu(),'omega':omega.detach().cpu()}
    functions={};jets_cache={};graph_cache={}
    def energy(z,c):return model(z[:m][None],z[m:][None],c[None])[0]
    grad=torch.func.grad(energy,argnums=0)
    def one(z,c):
        g,pullback=torch.func.vjp(lambda zz:grad(zz,c),z)
        action=torch.func.vmap(lambda v:pullback(v)[0],in_dims=1,out_dims=1)
        diagonal=(signs*action(signs)).mean(1)
        basis=torch.linalg.qr(action(omega)-diagonal[:,None]*omega,mode='reduced').Q
        core=basis.T@(action(basis)-diagonal[:,None]*basis)
        core=(core+core.T)*.5
        return g,diagonal,basis,core
    vectorized=torch.func.vmap(one,in_dims=(0,0))
    if batch_ad:
        def total_energy(z,c):return model(z[:,:m],z[:,m:],c).sum()
        batch_gradient=torch.func.grad(total_energy,argnums=0)
        def batch_construct(z,c):
            g,pullback=torch.func.vjp(lambda zz:batch_gradient(zz,c),z)
            def action(vectors):
                return torch.func.vmap(lambda v:pullback(v)[0])(vectors.permute(2,0,1)).permute(1,2,0)
            sg=signs[None].expand(len(z),-1,-1);om=omega[None].expand(len(z),-1,-1)
            if joint_probes:
                initial=action(torch.cat((sg,om),-1));hs,ho=initial[...,:probes],initial[...,probes:]
            else:hs,ho=action(sg),action(om)
            diagonal=(sg*hs).mean(-1)
            basis=torch.linalg.qr(ho-diagonal[...,None]*om,mode='reduced').Q
            core=basis.transpose(-1,-2)@(action(basis)-diagonal[...,None]*basis)
            return g,diagonal,basis,(core+core.transpose(-1,-2))*.5
        vectorized=batch_construct
    def build(generator,anchor,attrs,*,attr_scale,step_size,**kwargs):
        q,p=anchor.source_q,anchor.target_p
        signature=(q.data_ptr(),q._version,p.data_ptr(),p._version,attrs.data_ptr(),attrs._version,step_size)
        if cache and signature in jets_cache:
            runtime.stats['lowrank_cache_hits']=runtime.stats.get('lowrank_cache_hits',0)+1
            entry=jets_cache[signature]
            if hasattr(runtime,'h_stream') and not runtime.prefetching:
                torch.cuda.current_stream().wait_stream(runtime.h_stream)
                if not bool(entry[-1]):raise RuntimeError('low-rank jet nonfinite or B solve failed')
                entry[0].matrix.record_stream(torch.cuda.current_stream())
                entry[0].offset.record_stream(torch.cuda.current_stream())
            return entry[0]
        b,t=q.shape[:2]
        z=torch.cat((q.reshape(b*t,m),p.reshape(b*t,m)),-1)
        c=training.normalise_object_context(attrs,attr_scale)
        c=c[:,None].expand(b,t,*c.shape[1:]).reshape(b*t,*c.shape[1:])
        def construct(zz_all,cc_all):
            pieces=[]
            for start in range(0,b*t,chunk):
                zz,cc=zz_all[start:start+chunk],cc_all[start:start+chunk]
                key=len(zz)
                if key not in functions:
                    from torch.fx.experimental.proxy_tensor import make_fx
                    fx=make_fx(vectorized)(zz,cc)
                    functions[key]=(torch.compile(fx,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False}) if compiled else fx)
                pieces.append(functions[key](zz,cc))
            g,diagonal,basis,core=[torch.cat([v[i] for v in pieces],0) for i in range(4)]
            u,v=basis[:,:m],basis[:,m:]
            small=torch.eye(r,device=z.device,dtype=z.dtype)[None]+step_size*(v.transpose(-1,-2)@u)@core
            k,info=torch.linalg.solve_ex(small,v.transpose(-1,-2),check_errors=False)
            packed=torch.cat((zz_all,g,diagonal,core.flatten(1),k.flatten(1)),1)
            valid=(info==0).all() & torch.isfinite(packed).all() & torch.isfinite(basis).all()
            return basis,packed,valid
        if jet_graph:
            key=(b,t,step_size)
            if key not in graph_cache:
                za,ca=z.clone(),c.clone()
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):construct(za,ca)
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):outputs=construct(za,ca)
                graph_cache[key]=(za,ca,graph,outputs)
            za,ca,graph,outputs=graph_cache[key]
            za.copy_(z);ca.copy_(c);graph.replay()
            basis,packed,valid=outputs
            basis=basis.clone();packed=packed.clone()
        else:basis,packed,valid=construct(z,c)
        if not runtime.prefetching and not bool(valid):raise RuntimeError('low-rank jet nonfinite or B solve failed')
        runtime.stats['lowrank_rank']=r;runtime.stats['diagonal_probes']=probes
        runtime.lowrank_step=step_size
        result=HamiBallsAffineJets(matrix=basis.reshape(b,t,2*m,r),offset=packed.reshape(b,t,-1),health=None)
        if cache:
            jets_cache[signature]=(result,q,p,attrs,valid.clone())
            if len(jets_cache)>3:jets_cache.pop(next(iter(jets_cache)))
        return result
    def candidate(basis,packed,previous,attrs):
        n=previous.shape[1];q=previous[...,:2].reshape(-1,m);p=previous[...,2:].reshape(-1,m)
        z,g,diagonal=packed[:,:2*m],packed[:,2*m:4*m],packed[:,4*m:6*m]
        core=packed[:,6*m:6*m+r*r].reshape(-1,r,r)
        k=packed[:,6*m+r*r:].reshape(-1,r,m)
        u,v=basis[:,:m],basis[:,m:];h=runtime.lowrank_step
        mv=lambda A,x:torch.matmul(A,x.unsqueeze(-1)).squeeze(-1)
        dq=q-z[:,:m]
        rhs=p-z[:,m:]-h*g[:,:m]-h*(diagonal[:,:m]*dq+mv(u,mv(core,mv(u.transpose(-1,-2),dq))))
        dp=rhs-h*mv(u,mv(core,mv(k,rhs)))
        outq=z[:,:m]+h*g[:,m:]+dq+h*(diagonal[:,m:]*dp+mv(v,mv(core,mv(u.transpose(-1,-2),dq)+mv(v.transpose(-1,-2),dp))))
        outp=z[:,m:]+dp
        return torch.cat((outq.reshape(-1,n,2),outp.reshape(-1,n,2)),-1)
    original=training.learned_hamiballs_affine_jets
    if install_symbols:
        for module in tuple(sys.modules.values()):
            if module is not None and getattr(module,'learned_hamiballs_affine_jets',None) is original:
                module.learned_hamiballs_affine_jets=build
    runtime.integrator_candidate=candidate
    if hasattr(runtime,'h_stream'):
        if not cache:raise ValueError('Overlapped low-rank jets require identity cache')
        runtime.cached_jets=build
    runtime.stats['health_policy']='finite factors and nonsingular reduced solve'
    return build
