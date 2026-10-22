"""Split-local differentiations and separable weak integrals on observed fields."""
import json
import math
from pathlib import Path
import numpy as np
from numpy.polynomial import Polynomial
import torch
from .expressions import is_diff

def compact_weights(coords, width, count, max_order):
    n=len(coords)
    if n==1:return [np.ones((1,1))]+[np.zeros((1,1)) for _ in range(max_order)]
    width=min(max(width,11), n if n%2 else n-1)
    if width<9:raise ValueError('At least nine observations per differentiated axis required')
    half=width//2;centers=np.unique(np.linspace(half,n-half-1,count).astype(int))
    spacing=float(coords[1]-coords[0]);z=np.linspace(-1,1,width)
    poly=Polynomial([1,0,-1])**6
    g,gw=np.polynomial.legendre.leggauss(12)
    weights=np.zeros((max_order+1,width))
    for interval in range(width-1):
        points=(z[interval+1]+z[interval])/2+(z[interval+1]-z[interval])/2*g
        qw=(z[interval+1]-z[interval])/2*gw
        first=max(0,min(interval-3,width-9));inds=np.arange(first,first+9)
        basis=np.ones((len(points),9))
        for j,index in enumerate(inds):
            for other in inds:
                if other!=index:basis[:,j]*=(points-z[other])/(z[index]-z[other])
        for k in range(max_order+1):weights[k,inds]+=((qw*poly.deriv(k)(points))[:,None]*basis).sum(0)
    mats=[]
    for k in range(max_order+1):
        mat=np.zeros((len(centers),n));local=weights[k]/weights[0].sum()/(half*spacing)**k
        for i,c in enumerate(centers):mat[i,c-half:c+half+1]=local
        mats.append(mat)
    return mats

def periodic_weights(coords,kmax,max_order):
    n=len(coords);L=float(coords[1]-coords[0])*n;xx=coords-coords[0]
    mats=[]
    for order in range(max_order+1):
        rows=[np.ones(n) if order==0 else np.zeros(n)]
        for k in range(1,min(kmax,n//2-1)+1):
            phase=2*np.pi*k*xx/L+order*np.pi/2
            factor=(2*np.pi*k/L)**order
            rows.extend((factor*np.cos(phase),factor*np.sin(phase)))
        mats.append(np.array(rows)/n)
    return mats

def load_blocks(folder,device,manifold=True,energy=1-1e-12,kmax=4,weak_rows=0,
                weak_rows_list=None,weak_row_seed=0,weak_width_list=None):
    folder=Path(folder);meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    npz=np.load(folder/'observations.npz');axes=meta['spatial_axes']
    fields=[f['name'] for f in meta['fields']];blocks=[];audit=[]
    for split in ('train','validation'):
        desc=meta['splits'][split]
        trajectories=desc.get('trajectories',range(npz['field_'+fields[0]].shape[0]))
        bounds=desc.get('time_indices',[0,len(npz['t'])]);sl=slice(*bounds)
        for trajectory in trajectories:
            arrays={name:np.array(npz['field_'+name][trajectory,sl],dtype=np.float64) for name in fields}
            observation_mask = None
            if 'observation_mask' in npz.files:
                observation_mask = np.array(npz['observation_mask'][trajectory,sl],dtype=np.float64)
                expected_shape = next(iter(arrays.values())).shape
                if observation_mask.shape != expected_shape:
                    raise ValueError(f'observation_mask shape {observation_mask.shape} != field shape {expected_shape}')
            coords={'t':np.array(npz['t'][sl]),**{a:np.array(npz[a]) for a in axes}}
            if manifold and split=='train':
                for name,value in arrays.items():
                    q=torch.as_tensor(value.reshape(len(value),-1),device=device,dtype=torch.float64)
                    # Snapshot Gram eigensystem is small even for volumetric fields.
                    gram=q@q.T
                    try:
                        ev,U=torch.linalg.eigh(gram)
                        backend='torch_eigh'
                    except torch.linalg.LinAlgError:
                        # Static material fields give repeated zero eigenvalues.
                        # The small CPU Gram solve preserves the same projection.
                        values,vectors=np.linalg.eigh(gram.cpu().numpy())
                        ev=torch.as_tensor(values,device=device);U=torch.as_tensor(vectors,device=device)
                        backend='numpy_eigh_fallback'
                    ev=ev.clamp_min(0).flip(0);U=U.flip(1)
                    if ev.sum()>0:
                        rank=min(int(torch.searchsorted(torch.cumsum(ev,0)/ev.sum(),energy))+1,len(ev))
                        projected=U[:,:rank]@(U[:,:rank].T@q)
                        error=float(torch.linalg.norm(projected-q)/torch.linalg.norm(q))
                        arrays[name]=projected.cpu().numpy().reshape(value.shape)
                        audit.append({'field':name,'trajectory':int(trajectory),'rank':rank,'snapshots':len(value),'relative_projection_error':error,'eigensolver':backend})
            arrays['1']=np.ones_like(next(iter(arrays.values())))
            row_scales = list(dict.fromkeys(int(s) for s in (weak_rows_list or []) or [weak_rows]))
            width_scales = [int(s) for s in (weak_width_list or [])]
            if not width_scales:
                width_scales = [None] * len(row_scales)
            elif len(width_scales) == 1 and len(row_scales) > 1:
                width_scales = width_scales * len(row_scales)
            elif len(width_scales) != len(row_scales):
                raise ValueError('weak_width_list must have one width or one width per weak-row scale')
            for scale, width in zip(row_scales, width_scales):
                block=Block(arrays,coords,axes,meta['boundary'],device,split,kmax,scale,
                            observation_mask=observation_mask,
                            weak_row_seed=weak_row_seed,weak_width=width)
                block.observation_metadata=meta
                blocks.append(block)
    return meta,blocks,audit

class Block:
    def __init__(self,fields,coords,axes,boundary,device,split,kmax=4,weak_rows=0,
                 observation_mask=None,weak_row_seed=0,weak_width=None):
        self.observation_metadata={}
        self.coords=coords;self.axes=['t',*axes];self.shape=next(iter(fields.values())).shape
        self.device=device;self.split=split;self.boundary=boundary;self.stencils={}
        self.observation_mask = (torch.as_tensor(observation_mask.reshape(-1,1),device=device,
                               dtype=torch.float64) if observation_mask is not None else None)
        self.observation_fraction = (float(observation_mask.mean())
                                     if observation_mask is not None else 1.0)
        self.full_observation = observation_mask is None or bool(np.all(observation_mask == 1))
        self.fields={k:torch.as_tensor(v.reshape(-1,1),device=device,dtype=torch.float64) for k,v in fields.items()}
        self.weights=[]
        for axis,coord in coords.items():
            if axis=='t':ww=compact_weights(coord,31,min(8,max(3,len(coord)//4)),2)
            elif len(axes)==1 or boundary.get(axis)!='periodic':
                default_width=max(13,int(.3*len(coord))//2*2+1)
                ww=compact_weights(coord,weak_width if weak_width is not None else default_width,
                                   24 if len(axes)==1 else 6,4)
            else:ww=periodic_weights(coord,kmax,4)
            self.weights.append([torch.as_tensor(w,device=device,dtype=torch.float64) for w in ww])
        self.full_rows=int(np.prod([w[0].shape[0] for w in self.weights]))
        self.row_indices=None
        if weak_rows and self.full_rows>weak_rows:
            # Fixed stratified quadrature selection depends only on dimensions.
            nt=len(self.weights[0][0]);ns=self.full_rows//nt
            rng=np.random.default_rng(int(weak_row_seed))
            nspace=max(1,weak_rows//nt);space=np.sort(rng.choice(ns,nspace,replace=False))
            # Preserve the spatial zero mode when the spatial basis is Fourier.
            # Replace one selected row so the row budget and sorted order stay fixed.
            if len(axes)>1 and all(boundary.get(a)=='periodic' for a in axes) and 0 not in space:
                space=np.sort(np.concatenate(([0],space[1:])))
            self.row_indices=np.concatenate([space+i*ns for i in range(nt)])
        self.selected_indices=None
        self.observed_zero_weight=None
        self.observed_coverage=None
        self.observed_support_count=None
        if self.observation_mask is not None:
            # One dimensionless zero-order coverage is shared by every
            # differential order. Derivative-weight sums have physical units
            # and must not be used to normalize derivative columns.
            mask=self.observation_mask.reshape(*self.shape)
            denom=mask;full=mask.new_ones(self.shape)
            observed=(mask>0).to(mask.dtype);support_full=mask.new_ones(self.shape)
            for axis in range(len(self.axes)):
                w=torch.abs(self.weights[axis][0])
                denom=torch.tensordot(w,denom,dims=([1],[axis])).movedim(0,axis)
                full=torch.tensordot(w,full,dims=([1],[axis])).movedim(0,axis)
                support=(w>torch.finfo(w.dtype).eps*w.amax(1,keepdim=True)).to(w.dtype)
                observed=torch.tensordot(support,observed,dims=([1],[axis])).movedim(0,axis)
                support_full=torch.tensordot(support,support_full,dims=([1],[axis])).movedim(0,axis)
            self.observed_zero_weight=denom.reshape(self.full_rows,1)
            self.observed_coverage=self.observed_zero_weight/full.reshape(self.full_rows,1).clamp_min(1e-30)
            self.observed_support_count=observed.reshape(-1)
            min_support=torch.minimum(support_full.reshape(-1),observed.new_tensor(9.0))
            valid=(self.observed_coverage[:,0]>=max(0.25*self.observation_fraction,1e-3)) & (self.observed_support_count>=min_support)
            base=(torch.as_tensor(self.row_indices,device=device,dtype=torch.long)
                  if self.row_indices is not None else torch.arange(self.full_rows,device=device))
            self.selected_indices=base[valid[base]]
            if self.selected_indices.numel()==0:
                raise ValueError('No weak windows contain enough observed points')
        self.rows=(int(self.selected_indices.numel()) if self.selected_indices is not None else
                   (len(self.row_indices) if self.row_indices is not None else self.full_rows))
    def derivative(self,value,letters):
        value=value.reshape(*self.shape,-1)
        for a in sorted(set(letters)):
            order=letters.count(a);axis=self.axes.index(a);n=self.shape[axis]
            dx=float(self.coords[a][1]-self.coords[a][0])
            if self.boundary.get(a)=='periodic':
                k=2*np.pi*torch.fft.rfftfreq(n,d=dx,device=value.device,dtype=value.dtype)
                factor=(1j*k)**order
                if order%2 and n%2==0:factor[-1]=0
                shape=[1]*value.ndim;shape[axis]=len(k)
                value=torch.fft.irfft(torch.fft.rfft(value,dim=axis)*factor.reshape(shape),n=n,dim=axis)
            else:
                key=(a,order)
                if key not in self.stencils:
                    width=min(11,n);mat=np.zeros((n,n))
                    for i in range(n):
                        inds=np.arange(max(0,min(i-width//2,n-width)),max(0,min(i-width//2,n-width))+width)
                        offsets=(inds-i).astype(float);target=np.zeros(width);target[order]=math.factorial(order)
                        mat[i,inds]=np.linalg.solve(np.array([offsets**j for j in range(width)]),target)/dx**order
                    self.stencils[key]=torch.as_tensor(mat,device=value.device,dtype=value.dtype)
                value=torch.tensordot(self.stencils[key],value,dims=([1],[axis])).movedim(0,axis)
        return value.reshape(-1,value.shape[-1])
    def project(self,value,letters='',time_order=0):
        q=value.reshape(*self.shape,-1)
        if self.observation_mask is not None and not self.full_observation:
            mask=self.observation_mask.reshape(*self.shape,1)
            # Hidden NaNs are removed before multiplication. NaN*0 remains NaN.
            q=torch.where(mask>0,q,torch.zeros((),device=q.device,dtype=q.dtype))*mask
            integral=q
            derivative_mask=mask
            mean_numerator=q
            differential=bool(letters or time_order)
            for axis,name in enumerate(self.axes):
                order=time_order if name=='t' else letters.count(name)
                w=self.weights[axis][order]*((-1)**order)
                integral=torch.tensordot(w,integral,dims=([1],[axis])).movedim(0,axis)
                if differential:
                    derivative_mask=torch.tensordot(w,derivative_mask,dims=([1],[axis])).movedim(0,axis)
                    positive=torch.abs(self.weights[axis][0])
                    mean_numerator=torch.tensordot(positive,mean_numerator,dims=([1],[axis])).movedim(0,axis)
            integral=integral.reshape(self.full_rows,-1)
            if differential:
                # Center constants with an observed, positive quadrature mean.
                # This preserves zero response to a constant on a partial mask.
                mean=mean_numerator.reshape(self.full_rows,-1)/self.observed_zero_weight.clamp_min(1e-30)
                integral=integral-mean*derivative_mask.reshape(self.full_rows,1)
            integral=integral/self.observed_coverage.clamp_min(1e-30)
            return integral[self.selected_indices]
        for axis,name in enumerate(self.axes):
            order=time_order if name=='t' else letters.count(name)
            if order:q=q-q.select(axis,0).unsqueeze(axis)
            w=self.weights[axis][order]*((-1)**order)
            q=torch.tensordot(w,q,dims=([1],[axis])).movedim(0,axis)
        q=q.reshape(self.full_rows,-1)
        if self.selected_indices is not None:return q[self.selected_indices]
        return q if self.row_indices is None else q[self.row_indices]

class Evaluator:
    def __init__(self,block):
        self.block=block;self.cache={};self.weak_cache={};self.hits=0;self.computed=0;self.projected=0
    def field(self,e):return self.batch([e])
    def batch(self,exprs):
        missing=list(dict.fromkeys(e for e in exprs if e not in self.cache));groups={}
        self.hits += len(exprs)-len(missing)
        for e in missing:
            if not e.args:self.cache[e]=self.block.fields[e.op];self.computed+=1
            else:groups.setdefault((e.depth,e.op),[]).append(e)
        for (_,op),members in sorted(groups.items()):
            members=[e for e in members if e not in self.cache]
            if not members:continue
            a=self.batch([e.args[0] for e in members])
            if op in ('Add','Mul','Div'):
                b=self.batch([e.args[1] for e in members]);v={'Add':torch.add,'Mul':torch.mul,'Div':torch.div}[op](a,b)
            elif is_diff(op):v=self.block.derivative(a,op[1:])
            else:v={'Sin':torch.sin,'Cos':torch.cos,'Tanh':torch.tanh}[op](a)
            for j,e in enumerate(members):self.cache[e]=v[:,j:j+1]
            self.computed+=len(members)
        return torch.cat([self.cache[e].to(self.block.device) for e in exprs],1)

    def transient_batch(self, exprs):
        """Evaluate a candidate batch without retaining candidate-sized tensors.

        Search-layer candidates only read the frozen prefix cache.  Keeping every
        transient column in ``self.cache`` creates thousands of Python objects and
        forces the prefix cache to be rebuilt after every batch.  This path keeps
        the same tensor operations while returning the batch as one dense tensor.
        """
        if not exprs:
            return torch.empty((int(np.prod(self.block.shape)), 0),
                               device=self.block.device, dtype=torch.float64)
        if all(e in self.cache for e in exprs):
            return torch.cat([self.cache[e].to(self.block.device) for e in exprs], 1)
        signature={(e.depth,e.op) for e in exprs if e not in self.cache}
        if len(signature)==1 and all(e not in self.cache for e in exprs):
            op=exprs[0].op
            if not exprs[0].args:
                return torch.cat([self.block.fields[e.op] for e in exprs],1)
            a=self.transient_batch([e.args[0] for e in exprs])
            if op in ('Add','Mul','Div'):
                b=self.transient_batch([e.args[1] for e in exprs])
                return {'Add':torch.add,'Mul':torch.mul,'Div':torch.div}[op](a,b)
            if is_diff(op):return self.block.derivative(a,op[1:])
            return {'Sin':torch.sin,'Cos':torch.cos,'Tanh':torch.tanh}[op](a)
        columns=[None]*len(exprs);groups={}
        for i,e in enumerate(exprs):
            if e in self.cache:columns[i]=self.cache[e].to(self.block.device)
            else:groups.setdefault((e.depth,e.op),[]).append((i,e))
        for members in groups.values():
            values=self.transient_batch([e for _,e in members])
            for j,(i,_) in enumerate(members):columns[i]=values[:,j:j+1]
        return torch.cat(columns,1)
    def weak(self,exprs):
        def plan(e,letters=''):
            while is_diff(e.op):letters+=e.op[1:];e=e.args[0]
            if e.op=='Add':return plan(e.args[0],letters)+plan(e.args[1],letters)
            return [(e,''.join(sorted(letters)))]
        plans=[plan(e) for e in exprs];groups={}
        for e,letters in dict.fromkeys(k for p in plans for k in p):
            if (e,letters) not in self.weak_cache:groups.setdefault(letters,[]).append(e)
        for letters,members in groups.items():
            v=self.block.project(self.batch(members),letters)
            for j,e in enumerate(members):self.weak_cache[(e,letters)]=v[:,j:j+1]
            self.projected+=len(members)
        return torch.cat([sum(self.weak_cache[k] for k in p) for p in plans],1)

    @staticmethod
    def compile_weak_stream_plan(exprs):
        """Compile the expression layout once for all block evaluators.

        The plan contains only expression identities and integer gather layouts;
        block values are still computed independently by each evaluator.
        """
        def plan(e,letters=''):
            while is_diff(e.op):letters+=e.op[1:];e=e.args[0]
            if e.op=='Add':return plan(e.args[0],letters)+plan(e.args[1],letters)
            return [(e,''.join(sorted(letters)))]
        plans=[plan(e) for e in exprs]
        keys=list(dict.fromkeys(k for p in plans for k in p))
        key_index={key:i for i,key in enumerate(keys)}
        groups={}
        for i,key in enumerate(keys):groups.setdefault(key[1],[]).append((i,key[0]))
        layouts=[]
        by_length={}
        for i,p in enumerate(plans):by_length.setdefault(len(p),[]).append(i)
        for length,indices in by_length.items():
            gather=torch.as_tensor([[key_index[k] for k in plans[i]] for i in indices],dtype=torch.long)
            layouts.append((length,tuple(indices),gather))
        return {'plans':plans,'keys':keys,'groups':tuple(groups.items()),'layouts':tuple(layouts)}

    def weak_stream(self, exprs=None, compiled_plan=None):
        """Project a transient batch while preserving shared-prefix reuse."""
        compiled_plan = compiled_plan or self.compile_weak_stream_plan(exprs)
        plans=compiled_plan['plans'];keys=compiled_plan['keys']
        values=[None]*len(keys)
        for i,key in enumerate(keys):
            if key in self.weak_cache:values[i]=self.weak_cache[key].to(self.block.device)
        for letters,members in compiled_plan['groups']:
            pending=[(i,e) for i,e in members if values[i] is None]
            if not pending:continue
            projected=self.block.project(self.transient_batch([e for _,e in pending]),letters)
            for j,(i,_) in enumerate(pending):values[i]=projected[:,j:j+1]
            self.projected+=len(pending)
        base=torch.cat(values,1)
        out=torch.empty((base.shape[0],len(plans)),device=base.device,dtype=base.dtype)
        for length,indices,gather in compiled_plan['layouts']:
            gather=gather.to(device=base.device)
            selected=base[:,gather]
            out[:,torch.as_tensor(indices,device=base.device)]=selected.sum(-1) if length>1 else selected[...,0]
        return out
    def weak_direct(self,exprs):
        groups={};out=[None]*len(exprs)
        for i,e in enumerate(exprs):
            letters=''
            while is_diff(e.op):letters+=e.op[1:];e=e.args[0]
            groups.setdefault(''.join(sorted(letters)),[]).append((i,e))
        for letters,members in groups.items():
            v=self.block.project(self.batch([e for _,e in members]),letters)
            for j,(index,_) in enumerate(members):out[index]=v[:,j:j+1]
        return torch.cat(out,1)
    @staticmethod
    def _compact(value):
        storage_bytes=value.untyped_storage().nbytes()
        logical_bytes=value.numel()*value.element_size()
        return value.clone() if storage_bytes>logical_bytes else value
    def retain(self,exprs,compact=True):
        wanted=set()
        def visit(e):
            if e in wanted:return
            wanted.add(e)
            for a in e.args:visit(a)
        for e in exprs:visit(e)
        transform=self._compact if compact else (lambda value:value)
        self.cache={e:transform(v) for e,v in self.cache.items() if e in wanted}
        self.weak_cache={k:transform(v) for k,v in self.weak_cache.items() if k[0] in wanted}
    def offload(self):
        self.cache={e:v.cpu() for e,v in self.cache.items()}
    def prune_to(self,permanent_nodes):
        self.retain(permanent_nodes,compact=False)
    def release_values(self):
        self.cache.clear();self.weak_cache.clear()
