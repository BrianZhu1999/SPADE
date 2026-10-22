"""Uniform clean-field search. Ground-truth equations are never imported."""
import argparse
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import sympy as sp
from core.expressions import Expr, Grammar, Algebra, unpack, make, is_diff
from core.fields import load_blocks, Evaluator
from core.evaluation_factory import create_evaluator, algebra_class, create_spatial_guard
from core.conservation import audit_terms, is_flux_expression
from core.scheduler import Scheduler
from core.pareto import knee
from core import conditional_selector as selector

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def out_json(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def targets(meta,evaluators):
    cols=[]
    for target in meta['targets']:
        expr=unpack(target['expression']) if 'expression' in target else Expr(target['field'])
        if 'multiplier' in target:expr=Expr('Mul',(Expr(target['multiplier']),expr))
        cols.append(torch.cat([e.block.project(e.field(expr),time_order=target['time_order']) for e in evaluators]))
    return torch.cat(cols,1)

def rank_prefix(exprs,table,target,ntrain,cap):
    if len(exprs)<=cap:return list(range(len(exprs)))
    z=table[:ntrain];y=target[:ntrain]
    scale=np.sqrt(np.einsum('ij,ij->j',z,z));dots=z.T@y
    score=np.max(np.abs(dots)/(scale[:,None]*np.maximum(np.linalg.norm(y,axis=0),1e-30)),axis=1)
    # A flux can be weakly correlated with a time derivative while its outer
    # spatial derivative is strongly correlated. Retain that measured context.
    positions={e:i for i,e in enumerate(exprs)}
    for i,e in enumerate(exprs):
        if is_diff(e.op) and e.args[0] in positions:
            parent=positions[e.args[0]]
            score[parent]=max(score[parent],score[i])
    by={}
    for i,e in enumerate(exprs):by.setdefault((e.op,e.size),[]).append(i)
    keep=set(i for i,e in enumerate(exprs) if e.size<=2 or e.depth<=1)
    # Equal quotas preserve weakly correlated operator families and small trees.
    quota=max(2,cap//max(1,len(by)))
    for inds in by.values():
        for j in sorted(inds,key=lambda i:-score[i])[:quota]:keep.add(j)
    if len(keep)>cap:
        keep=set(sorted(keep,key=lambda i:(exprs[i].size,-score[i]))[:cap])
    for i in np.argsort(-score):
        if len(keep)>=cap:break
        keep.add(int(i))
    return sorted(keep,key=lambda i:(exprs[i].size,exprs[i].text()))


def differential_closure(expressions,grammar,seen):
    """Outer derivative look-ahead before correlation-based prefix pruning.

    Every field and operator is treated alike. Traversal uses mathematical
    operation count, including the order of a differential node, then text.
    No reference equations, variable names, target units or fitted coefficients
    enter the traversal.
    """
    def cost(e):
        return (len(e.op)-1 if is_diff(e.op) else 1)+sum(cost(c) for c in e.args)
    operations=[op for op in grammar.operations() if is_diff(op)]
    for e in sorted(expressions,key=lambda e:(cost(e),e.text())):
        for op in operations:
            candidate=make(op,e)
            if not grammar.allowed(candidate):continue
            if candidate in seen:continue
            seen.add(candidate)
            yield candidate

def validation_groups(blocks):
    valid=[b for b in blocks if b.split=='validation']
    if len(valid)>1:return np.concatenate([np.full(b.rows,i) for i,b in enumerate(valid)])
    groups=[];group_id=0
    for block in valid:
        time_weights=block.weights[0][0].cpu().numpy()
        nspace=block.full_rows//len(time_weights);last=-1;local_groups=[]
        for row in time_weights:
            occupied=np.flatnonzero(np.abs(row)>1e-15)
            if len(occupied) and occupied[0]>last:
                local_groups.extend([group_id]*nspace);group_id+=1;last=int(occupied[-1])
            else:local_groups.extend([-1]*nspace)
        if block.row_indices is not None:local_groups=np.array(local_groups)[block.row_indices].tolist()
        groups.extend(local_groups)
    return np.array(groups)


def pareto_select(entries,table,target,exprs,ntrain,mode,groups=None):
    # A candidate with a non-finite validation score cannot participate in
    # model selection.  Keeping it would make min() choose an undefined score
    # and could silently write an invalid result for noisy observations.
    entries=[row for row in entries if all(np.isfinite(float(row.get(key, np.nan)))
                                          for key in ('training_relative_mse','validation_relative_mse','score'))]
    if not entries:
        raise RuntimeError('No finite validation candidates available for model selection')
    best={}
    for row in entries:
        c=row['complexity']
        if c not in best or row['validation_relative_mse']<best[c]['validation_relative_mse']:best[c]=row
    front=[];error=float('inf')
    for complexity in sorted(best):
        row=best[complexity]
        if row['validation_relative_mse']<error:front.append(row);error=row['validation_relative_mse']
    if mode=='minimum_residual':chosen=min(front,key=lambda r:r['validation_relative_mse'])
    elif mode=='penalized_validation':chosen=min(front,key=lambda r:(r['score'],r['complexity'],r['validation_relative_mse']))
    elif mode=='knee':
        chosen,diagnostic=knee(front)
        chosen['pareto_geometry']=diagnostic
    else:
        top=min(front,key=lambda r:r['validation_relative_mse'])
        index={e.text():i for i,e in enumerate(exprs)}
        pred=sum(term['coefficient']*table[ntrain:,index[term['text']]] for term in top['terms'])
        losses=(pred-target[ntrain:])**2/max(np.mean(target[ntrain:]**2),1e-30)
        block_losses=[float(losses[groups==g].mean()) for g in np.unique(groups) if g>=0] if groups is not None else []
        se=float(np.std(block_losses,ddof=1)/np.sqrt(len(block_losses))) if len(block_losses)>1 else 0
        threshold=top['validation_relative_mse']+max(se,1e-8)
        chosen=min((r for r in front if r['validation_relative_mse']<=threshold),key=lambda r:(r['complexity'],r['validation_relative_mse']))
        chosen['validation_independent_groups']=len(block_losses)
        chosen['selection_tolerance']=max(se,1e-8)
    for row in front:row.pop('selected',None)
    chosen['selected']=True
    return chosen,front

def has_spatial_derivative(expression):
    """Return whether an expression contains a spatial differential node.

    This is a generic PDE-type guard for static targets.  It does not inspect
    target names, reference supports, coefficients, or test values.
    """
    return is_diff(expression.op) or any(
        has_spatial_derivative(child) for child in expression.args
    )

def unpack_selected_terms(row):
    return [(unpack(term['expression']), float(term['coefficient'])) for term in row['terms']]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default=str(Path(__file__).resolve().parent));p.add_argument('--case',required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--device',default='cuda:0')
    p.add_argument('--variant',default='full',choices=['full','clean_core','no_manifold','no_dimensions','no_pareto','round_robin','random'])
    p.add_argument('--layers',type=int,default=3);p.add_argument('--prefix-cap',type=int,default=512)
    p.add_argument('--budget',type=int,default=500000);p.add_argument('--chunk',type=int,default=64)
    p.add_argument('--decision-batch',type=int,default=64)
    p.add_argument('--compute-batch',type=int,default=0,help='Zero selects a device-memory-aware batch')
    p.add_argument('--compute-batch-cap',type=int,default=8192)
    p.add_argument('--workspace-gib',type=float,default=10.0)
    p.add_argument('--closure-budget',type=int,default=100000)
    p.add_argument('--max-terms',type=int,default=8);p.add_argument('--seconds',type=int,default=14400)
    p.add_argument('--kmax',type=int,default=4);p.add_argument('--beam',type=int,default=64)
    p.add_argument('--swap-rounds',type=int,default=2)
    p.add_argument('--swap-beam',type=int,default=16)
    p.add_argument('--weak-rows',type=int,default=0,help='Maximum weak rows per trajectory, zero keeps all')
    p.add_argument('--weak-rows-list',default='',help='Comma-separated short/medium/long weak-row counts combined in one search')
    p.add_argument('--weak-width-list',default='',help='Comma-separated spatial test-function support widths paired with weak-row scales')
    p.add_argument('--weak-row-seed',type=int,default=0,help='Seed for dimension-only weak-row resampling')
    p.add_argument('--feature-cap',type=int,default=200000)
    p.add_argument('--save-full-features',action='store_true')
    p.add_argument('--budgeted-selection',action='store_true',
                   help='Use remaining wall time for selection and save branch checkpoints')
    p.add_argument('--run-id',default='development_v1')
    p.add_argument('--pareto-rule',choices=['knee','independent_standard_error','penalized_validation'],default='penalized_validation')
    p.add_argument('--complexity-penalty',type=float,default=1e-8)
    p.add_argument('--fit-method',choices=['svd','qr'],default='svd',
                   help='Stable coefficient factorization used during model selection')
    p.add_argument('--operator-prior', choices=['open','polynomial_differential','flux_conservation'], default='open',
                   help='Generic grammar. flux_conservation keeps open algebra inside an outer spatial derivative.')
    p.add_argument('--conservation-axis', choices=['x','y','z'], default=None,
                   help='Optional outer-flux direction filter; unset preserves multi-direction flux behavior')
    p.add_argument('--conservation-audit', action='store_true',
                   help='Record symbolic Euler-operator conservation audits for selected equations')
    p.add_argument('--robust-fit',action='store_true',
                   help='Use Huber iteratively reweighted fitting on training rows')
    p.add_argument('--huber-delta',type=float,default=1.5)
    p.add_argument('--irls-iterations',type=int,default=6)
    p.add_argument('--fast-final-selection',action='store_true',
                   help='Select from completed branch paths without repeated symbolic guard scans')
    p.add_argument('--require-spatial-term',action='store_true',
                   help='For static targets, retain only models containing at least one spatial derivative')
    p.add_argument('--enforce-target-units',action='store_true',
                   help='Keep candidate columns whose physical units match at least one target_units entry')
    p.add_argument('--final-reserve',type=float,default=30.0,
                   help='Seconds reserved for final validation and result writing')
    a=p.parse_args();root=Path(a.root);out=root/'results'/a.run_id/'SPADE'/a.case/a.variant/f'seed{a.seed}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError('Preserve completed result; choose new root/variant/seed')
    started=time.perf_counter();torch.set_num_threads(4);torch.set_grad_enabled(False)
    code_root=Path(__file__).resolve().parent
    source_hashes={str(p.relative_to(code_root)):digest(p) for p in [Path(__file__).resolve(),*sorted((code_root/'core').glob('*.py'))]}
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    torch.empty((),device=a.device);torch.cuda.reset_peak_memory_stats(a.device)
    use_manifold=a.variant not in ('no_manifold','clean_core')
    use_dimensions=a.variant not in ('no_dimensions','clean_core')
    weak_rows_list=[int(x) for x in a.weak_rows_list.split(',') if x.strip()]
    weak_width_list=[int(x) for x in a.weak_width_list.split(',') if x.strip()]
    meta,blocks,manifold=load_blocks(root/'data'/a.case,a.device,use_manifold,kmax=a.kmax,weak_rows=a.weak_rows,
                                     weak_rows_list=weak_rows_list or None,weak_row_seed=a.weak_row_seed,
                                     weak_width_list=weak_width_list or None)
    metadata_hash=digest(root/'data'/a.case/'metadata.json')
    out_json(out/'metadata_snapshot.json',meta)
    evals=[create_evaluator(b,meta) for b in blocks];ntrain=sum(b.rows for b in blocks if b.split=='train')
    y=targets(meta,evals);yn=y.cpu().numpy();val_groups=validation_groups(blocks)
    exclude={t['field'] for t in meta['targets'] if t['time_order']==0 and 'field' in t}
    static=all(t['time_order']==0 for t in meta['targets'])
    states={f['name'] for f in meta['fields'] if f.get('role')=='state' and f['name'] not in exclude}
    def depends_state(e):return e.op in states or any(depends_state(c) for c in e.args)
    units={f['name']:f['units'] for f in meta['fields'] if f['name'] not in exclude}
    max_order = 2 if a.operator_prior == 'polynomial_differential' else 4
    grammar=Grammar(units,meta['spatial_axes'],meta.get('coordinate_units'),use_dimensions,
                    max_order=max_order, operator_prior=a.operator_prior)
    pool=[Expr('1')]+[Expr(name) for name in units]
    observed_values=sum(int(np.prod(b.shape)) for b in blocks)
    # The CPU stores a large shared DAG when full field values exceed VRAM.
    offload=observed_values*a.prefix_cap*8>8*1024**3
    memory_prefix_cap=(a.prefix_cap if a.variant=='clean_core' else
                       max(len(pool),min(a.prefix_cap,int(28*1024**3/max(observed_values*16,1)))))
    max_block_values=max(int(np.prod(b.shape)) for b in blocks)
    legacy_chunk=min(a.chunk,max(4,int(768*1024**2/max(max_block_values*8,1))))
    decision_batch=max(1,a.decision_batch)
    automatic_compute=max(decision_batch,min(a.compute_batch_cap,
        int(a.workspace_gib*1024**3/max(max_block_values*8*6,1))))
    compute_batch=max(1,a.compute_batch) if a.compute_batch else automatic_compute
    if a.variant!='clean_core':compute_batch=legacy_chunk
    mode=('round_robin' if a.variant=='clean_core' else
          (a.variant if a.variant in ('round_robin','random') else 'q_learning'))
    scheduler=Scheduler(a.seed,mode);seen=set();exprs=[];tables=[];stats=[];total=0
    ytrain,yval=y[:ntrain],y[ntrain:]
    ynorm=(ytrain.square().sum(0)).clamp_min(1e-30)
    best_gain=np.zeros(y.shape[1]);enumeration_started=time.perf_counter()
    for layer in range(1,a.layers+1):
        last_progress=time.perf_counter()
        for e in evals:
            for start in range(0,len(pool),compute_batch):
                e.batch(pool[start:start+compute_batch])
                if offload:e.offload()
        streams={op:iter(grammar.stream(pool,op,seen)) for op in grammar.operations()}
        if layer==1:streams['terminal']=iter(pool)
        exhausted=set();stage_expr=[];raw=0
        # Prefix stages also have the same computational cap, independent of case.
        stage_budget=a.budget if layer==a.layers else min(a.budget,100000)
        closure_budget=a.closure_budget if layer<a.layers else 0
        cache_dir=out/'enumeration_cache';cache_dir.mkdir(exist_ok=True)
        stage_store=np.lib.format.open_memmap(cache_dir/f'layer_{layer}.npy',mode='w+',dtype=np.float64,
            shape=(len(yn),stage_budget+closure_budget),fortran_order=True)
        out_json(cache_dir/f'prefix_{layer}.json',{'layer':layer,'expressions':[e.pack() for e in pool]})
        deterministic_batches=a.variant=='clean_core'
        while raw<stage_budget and len(exhausted)<len(streams):
            if time.perf_counter()-started>a.seconds*.65:break
            candidates=[];scheduled=[]
            while (len(candidates)<compute_batch and raw+len(candidates)<stage_budget and
                   len(exhausted)<len(streams)):
                families=[f for f in streams if f not in exhausted]
                family=scheduler.choose(families,layer,(raw+len(candidates))/stage_budget)
                take=(min(decision_batch,compute_batch-len(candidates),stage_budget-raw-len(candidates))
                      if deterministic_batches else min(legacy_chunk,stage_budget-raw))
                part=list(itertools.islice(streams[family],take))
                if not part:
                    exhausted.add(family)
                    if not deterministic_batches:break
                    continue
                candidates.extend(part);scheduled.append((family,len(part),families))
                if deterministic_batches:
                    scheduler.update(family,0.,len(part),families)
                else:break
            if not candidates:continue
            weak_plan=Evaluator.compile_weak_stream_plan(candidates)
            vals=torch.cat([e.weak_stream(candidates,weak_plan) for e in evals])
            good=torch.isfinite(vals).all(0)&(vals[:ntrain].abs().amax(0)<1e12)&(vals[:ntrain].square().sum(0)>1e-24)
            raw+=len(candidates);total+=len(candidates)
            indices=torch.where(good)[0]
            reward=0.
            if len(indices):
                kept=candidates if len(indices)==len(candidates) else [candidates[i] for i in indices.cpu().tolist()]
                valid=vals[:,indices]
                if not deterministic_batches:
                    coef=(valid[:ntrain].T@ytrain)/(valid[:ntrain].square().sum(0)[:,None].clamp_min(1e-30))
                    train_gain=(valid[:ntrain].T@ytrain).square()/(valid[:ntrain].square().sum(0)[:,None]*ynorm).clamp_min(1e-30)
                    best=torch.argmax(train_gain,dim=0)
                    preds=valid[ntrain:,best]*coef[best,torch.arange(y.shape[1],device=y.device)]
                    val_gain=1-((preds-yval).square().sum(0)/yval.square().sum(0).clamp_min(1e-30))
                    val_gain_cpu=val_gain.cpu().numpy()
                    improvement=np.maximum(val_gain_cpu-best_gain,0)
                    reward=float(improvement.mean());best_gain=np.maximum(best_gain,val_gain_cpu)
                begin=len(stage_expr);stage_store[:,begin:begin+len(kept)]=valid.cpu().numpy()
                stage_expr.extend(kept)
            if not deterministic_batches:
                family,_,families=scheduled[0]
                scheduler.update(family,reward,len(candidates),families)
            if time.perf_counter()-last_progress>30:
                print(json.dumps({'stage':'enumerate','layer':layer,'evaluated':raw,'retained':len(stage_expr),'seconds':time.perf_counter()-started}),flush=True)
                last_progress=time.perf_counter()
        closure_raw=0
        closure_stream=iter(differential_closure(stage_expr,grammar,seen))
        while closure_raw<closure_budget and time.perf_counter()-started<a.seconds*.65:
            candidates=list(itertools.islice(closure_stream,min(compute_batch,closure_budget-closure_raw)))
            if not candidates:break
            weak_plan=Evaluator.compile_weak_stream_plan(candidates)
            vals=torch.cat([e.weak_stream(candidates,weak_plan) for e in evals])
            good=torch.isfinite(vals).all(0)&(vals[:ntrain].abs().amax(0)<1e12)&(vals[:ntrain].square().sum(0)>1e-24)
            indices=torch.where(good)[0]
            if len(indices):
                kept=candidates if len(indices)==len(candidates) else [candidates[i] for i in indices.cpu().tolist()]
                begin=len(stage_expr)
                stage_store[:,begin:begin+len(kept)]=vals[:,indices].cpu().numpy()
                stage_expr.extend(kept)
            closure_raw+=len(candidates);total+=len(candidates)
            if time.perf_counter()-last_progress>30:
                print(json.dumps({'stage':'differential_closure','layer':layer,'evaluated':closure_raw,
                                  'retained':len(stage_expr),'seconds':time.perf_counter()-started}),flush=True)
                last_progress=time.perf_counter()
        if not stage_expr:raise RuntimeError('No finite nonzero features')
        stage_store.flush();table=stage_store[:,:len(stage_expr)]
        out_json(cache_dir/f'layer_{layer}.json',{'expressions':[e.pack() for e in stage_expr],'finite_columns':len(stage_expr)})
        exprs.extend(stage_expr);tables.append(table)
        stats.append({'layer':layer,'input_nodes':len(pool),'evaluated':raw+closure_raw,
                      'operator_stream_evaluated':raw,'differential_closure_evaluated':closure_raw,
                      'finite_columns':len(stage_expr),'stream_exhausted':len(exhausted)==len(streams),'grammar_counts':dict(grammar.counts)})
        print(json.dumps({'stage':'layer_done',**stats[-1]}),flush=True)
        if layer<a.layers:
            # Retain roots on training correlation, complexity and operator diversity.
            joint=pool+stage_expr
            old=torch.cat([e.weak(pool) for e in evals]).cpu().numpy()
            ptable=np.concatenate((old,table),1)
            ids=rank_prefix(joint,ptable,yn,ntrain,memory_prefix_cap)
            pool=list(dict.fromkeys(joint[i] for i in ids))
            for e in evals:e.retain(pool)
            del ptable,old
        if time.perf_counter()-started>a.seconds*.65:break
    rawtable=np.lib.format.open_memmap(cache_dir/'combined.npy',mode='w+',dtype=np.float64,
        shape=(len(yn),sum(t.shape[1] for t in tables)),fortran_order=True)
    offset=0
    for t in tables:
        rawtable[:,offset:offset+t.shape[1]]=t;offset+=t.shape[1]
    rawtable.flush();del tables,t,table,stage_store
    # First occurrence wins only for identical canonical AST, never by field similarity.
    unique={}
    for i,e in enumerate(exprs):unique.setdefault(e,i)
    ids=[i for e,i in unique.items() if not static or depends_state(e)]
    exprs=[exprs[i] for i in ids];rawtable=rawtable[:,ids]
    target_unit_filter=None
    if a.enforce_target_units:
        declared=meta.get('target_units')
        if not declared:
            raise ValueError('--enforce-target-units requires metadata.target_units')
        target_unit_filter={tuple(int(v) for v in unit) for unit in declared}
        keep=[i for i,e in enumerate(exprs) if grammar.dim(e) is not None and tuple(grammar.dim(e)) in target_unit_filter]
        if not keep:
            raise RuntimeError('No candidate columns match target_units')
        exprs=[exprs[i] for i in keep];rawtable=rawtable[:,keep]
    generated_finite_columns=len(exprs)
    if a.operator_prior == 'flux_conservation':
        keep=[i for i,e in enumerate(exprs) if is_flux_expression(e, a.conservation_axis)]
        if not keep:
            raise RuntimeError('No outer spatial-flux candidates retained')
        exprs=[exprs[i] for i in keep]
        rawtable=rawtable[:,keep]
    evaluated_columns=len(exprs)
    if len(exprs)>a.feature_cap:
        ids=rank_prefix(exprs,rawtable,yn,ntrain,a.feature_cap)
        exprs=[exprs[i] for i in ids];rawtable=rawtable[:,ids]
    enumeration_s=time.perf_counter()-enumeration_started
    np.save(cache_dir/'selection_table.npy',rawtable)
    np.save(cache_dir/'targets.npy',yn)
    np.save(cache_dir/'validation_groups.npy',val_groups)
    out_json(cache_dir/'selection_manifest.json',{'expressions':[e.pack() for e in exprs],
        'config':vars(a),'data_hash':meta['data_sha256'],'metadata_hash':metadata_hash,'source_hashes':source_hashes,
        'layers':stats,'ntrain':ntrain,'generated_evaluated':total,'finite_unique_columns':evaluated_columns,
        'enumeration_seconds':enumeration_s,'manifold':manifold})
    print(json.dumps({'stage':'selection','columns':len(exprs),'rows':len(rawtable),'ntrain':ntrain}),flush=True)
    shared_counts={'field_nodes_computed':sum(e.computed for e in evals),'field_cache_hits':sum(e.hits for e in evals),'weak_nodes_projected':sum(e.projected for e in evals),
                   'face_nodes_computed':sum(getattr(e,'face_nodes_computed',0) for e in evals),
                   'face_cache_hits':sum(getattr(e,'face_cache_hits',0) for e in evals)}
    spatial_guards=[create_spatial_guard(meta,evals,i) for i in range(len(meta['targets']))] if static else []
    for e in evals:
        if hasattr(e,'release_values'):e.release_values()
        else:e.cache.clear();e.weak_cache.clear()
    del evals,blocks,y
    gc.collect()
    torch.cuda.empty_cache()
    class SelectionAlgebra(algebra_class(meta)):
        def key(self,index):
            if static:
                value=self.normalized(self.exprs[index])
                state_functions=[sp.Function(s)(*self.coords) for s in states]
                if not value.has(*state_functions):return 'ZERO'
            return super().key(index)
    selector.Algebra=SelectionAlgebra;selector.BEAM_WIDTH=a.beam;selector.TOP_GLOBAL=128;selector.TOP_PER_SIZE=32;selector.PER_BUDGET=4
    selector.PENALTY=a.complexity_penalty
    def admissible_support(indices,coefs,branch_exprs,alg,target_index=0):
        if not static:return True
        value=sp.expand(sum(float(c)*alg.normalized(branch_exprs[i]) for i,c in zip(indices,coefs)))
        for term in sp.Add.make_args(value):
            coeff,body=term.as_coeff_Mul()
            if abs(float(coeff))<1e-10:continue
            if any(d.expr.has(*(sp.Function(s)(*alg.coords) for s in states)) for d in body.atoms(sp.Derivative)):
                return spatial_guards[target_index]([branch_exprs[i] for i in indices],coefs)
        return False
    # Parallel model families preserve simple algebraic paths when many
    # correlated transcendental approximants dominate a global beam.
    branches=[('algebraic_differential',[i for i,e in enumerate(exprs) if not (e.operators & {'Sin','Cos','Tanh','Div'})]),
              ('rational_differential',[i for i,e in enumerate(exprs) if not (e.operators & {'Sin','Cos','Tanh'})]),
              ('all_operators',list(range(len(exprs))))]
    branches=[row for i,row in enumerate(branches) if row[1] and not any(row[1]==old[1] for old in branches[:i])]
    paths=[[] for _ in meta['targets']]
    selection_deadline=started+a.seconds-max(10.0,float(a.final_reserve))
    completed_families=[]
    for bi,(name,ids) in enumerate(branches):
        if not ids:continue
        if a.budgeted_selection and time.perf_counter()>=selection_deadline:
            break
        branch_exprs=[exprs[i] for i in ids];branch_algebra=SelectionAlgebra(branch_exprs)
        # Validation ranks completed models, without pruning later families.
        complexity_bounds=None
        print(json.dumps({'stage':'model_family','family':name,'columns':len(ids)}),flush=True)
        now=time.perf_counter()
        branch_deadline=(now+max(0.0,(selection_deadline-now)/(len(branches)-bi))
                         if a.budgeted_selection else
                         now+max(120,(started+a.seconds-now)/(len(branches)-bi)))
        _,branch_paths=selector.select_equations(torch.from_numpy(rawtable[:,ids]),torch.from_numpy(yn),branch_exprs,ntrain,a.device,
            max_terms=a.max_terms,deadline=branch_deadline,max_condition=1e9,fit_method=a.fit_method,
            final_admissible=[(lambda support,coefs,qi=qi:admissible_support(support,coefs,branch_exprs,branch_algebra,qi))
                              for qi in range(len(meta['targets']))] if static else None,
            complexity_bounds=complexity_bounds,swap_rounds=a.swap_rounds,swap_beam=a.swap_beam,
            robust_fit=a.robust_fit,huber_delta=a.huber_delta,irls_iterations=a.irls_iterations)
        for qi,entries in enumerate(branch_paths):
            for entry in entries:entry['search_branch']=name
            paths[qi].extend(entries)
        completed_families.append(name)
        if a.budgeted_selection:
            checkpoint={
                'case':a.case,'method':'SPADE','run_id':a.run_id,'variant':a.variant,
                'seed':a.seed,'config':vars(a),'status':'branch_checkpoint',
                'data_hash':meta['data_sha256'],'metadata_hash':metadata_hash,
                'source_hashes':source_hashes,'targets':meta['targets'],
                'completed_families':list(completed_families),
                'selected':[min(rows,key=lambda r:(r['score'],r['complexity'],r['validation_relative_mse']))
                            if rows else None for rows in paths],
                'wall_seconds':time.perf_counter()-started,
                'enumeration_seconds':enumeration_s,'generated_evaluated':total,
                'selection_rule':a.pareto_rule,
            }
            pending_checkpoint=out/'branch_checkpoint.pending.json'
            out_json(pending_checkpoint,checkpoint)
            pending_checkpoint.replace(out/'branch_checkpoint.json')
    selected=[];fronts=[];spatial_guard_fallbacks=[]
    for i,entries in enumerate(paths):
        if static:
            algebra=SelectionAlgebra(exprs)
            if a.require_spatial_term:
                entries=[row for row in entries if any(
                    has_spatial_derivative(unpack(term['expression']))
                    for term in row['terms']
                )]
            def has_space_term(row):
                local=[unpack(term['expression']) for term in row['terms']]
                return admissible_support(range(len(local)),[term['coefficient'] for term in row['terms']],local,algebra,i)
            finite=[row for row in entries if all(np.isfinite(float(row.get(key, np.nan)))
                                                   for key in ('training_relative_mse', 'validation_relative_mse', 'score'))]
            if not a.fast_final_selection:
                # The spatial guard is a training-only numerical check, but
                # evaluating it for every catalogue entry can consume the
                # entire wall-clock budget after model search has finished.
                # Pareto candidates are the only rows that can be selected;
                # guard that bounded set under the same global deadline.
                best_by_complexity={}
                for row in finite:
                    c=row['complexity']
                    if c not in best_by_complexity or row['validation_relative_mse'] < best_by_complexity[c]['validation_relative_mse']:
                        best_by_complexity[c]=row
                guard_front=[];best_error=float('inf')
                for c in sorted(best_by_complexity):
                    row=best_by_complexity[c]
                    if row['validation_relative_mse'] < best_error:
                        guard_front.append(row);best_error=row['validation_relative_mse']
                guarded=[]
                for row in sorted(guard_front, key=lambda r:(r['score'], r['complexity'], r['validation_relative_mse'])):
                    if time.perf_counter() >= selection_deadline:
                        break
                    if has_space_term(row):
                        guarded.append(row)
                entries=guarded
            if not entries:
                # Noisy or sub-sampled static observations can make the
                # numerical contribution check unstable.  Keep the PDE type
                # constraint explicit by retaining only candidates containing
                # a spatial derivative, and record this fallback in the result.
                def contains_spatial_derivative(expr):
                    return has_spatial_derivative(expr)
                structural = [row for row in finite
                              if any(contains_spatial_derivative(unpack(term['expression']))
                                     for term in row['terms'])]
                if structural:
                    entries = structural
                    spatial_guard_fallbacks.append({'target': i, 'mode': 'structural_spatial_term',
                                                    'candidate_count': len(structural)})
                else:
                    raise RuntimeError('No spatial differential equation retained for steady observations')
        chosen,front=pareto_select(entries,rawtable,yn[:,i],exprs,ntrain,'minimum_residual' if a.variant=='no_pareto' else a.pareto_rule,val_groups)
        selected.append(chosen);fronts.append(front)
    conservation_audits=[]
    if a.conservation_audit:
        for target_index,row in enumerate(selected):
            target=meta['targets'][target_index]
            conservation_audits.append(audit_terms(
                unpack_selected_terms(row),
                state_name=target.get('field','u'),
                axis=a.conservation_axis or (meta.get('spatial_axes') or ['x'])[0]))
    result={'case':a.case,'method':'SPADE','run_id':a.run_id,'variant':a.variant,'seed':a.seed,'config':vars(a),'data_hash':meta['data_sha256'],
      'metadata_hash':metadata_hash,
      'targets':meta['targets'],'selected':selected,'pareto_fronts':fronts,'manifold':manifold,'layers':stats,
      'generated_evaluated':total,'finite_unique_columns':evaluated_columns,
      'generated_finite_columns':generated_finite_columns,'selection_columns':len(exprs),
      'resource_allocation':{'observed_values':observed_values,'prefix_cap_used':memory_prefix_cap,
                             'decision_batch_used':decision_batch,'compute_batch_used':compute_batch,
                             'legacy_chunk_used':legacy_chunk,'workspace_gib':a.workspace_gib,
                             'prefix_offload_to_host':offload},
      'shared_computation':shared_counts,
      'discretization':meta.get('discretization',{'type':'continuous_weak_projection'}),
      'spatial_contribution_checks':[{'checked':g.checked,'rejected':g.rejected,'last_report':g.last_report} for g in spatial_guards],
      'spatial_guard_fallbacks':spatial_guard_fallbacks,
      'conservation_audits':conservation_audits,
      'enumeration_seconds':enumeration_s,'wall_seconds':time.perf_counter()-started,
      'peak_gpu_bytes':torch.cuda.max_memory_allocated(a.device),'scheduler':{'mode':mode,'updates':len(scheduler.log),'positive_rewards':sum(x['reward']>0 for x in scheduler.log)},
      'source_hashes':source_hashes,
      'target_unit_filter':sorted([list(u) for u in target_unit_filter]) if target_unit_filter else None,
      'selection_rule':a.pareto_rule if a.variant!='no_pareto' else 'minimum_residual',
      'robust_fit':a.robust_fit,'huber_delta':a.huber_delta,'irls_iterations':a.irls_iterations,
      'historical_use':meta.get('historical_use','new_data')}
    if a.budgeted_selection:
        result['completed_families']=completed_families
        result['selection_termination']=('all_families_completed' if len(completed_families)==len(branches)
                                         else 'wall_budget_reached')
    out_json(out/'scheduler.json',scheduler.log)
    out_json(out/'candidates.json',{'expressions':[e.pack() for e in exprs],'ntrain':ntrain})
    selected_keys={t['text'] for front in fronts for row in front for t in row['terms']}
    saved_ids=list(range(len(exprs))) if a.save_full_features else [i for i,e in enumerate(exprs) if e.text() in selected_keys]
    np.savez_compressed(out/'weak_features.npz',table=rawtable[:,saved_ids],target=yn,indices=saved_ids)
    result['wall_seconds']=time.perf_counter()-started
    result['saved_feature_columns']=len(saved_ids)
    out_json(out/'result.json',result)
    print(json.dumps({'stage':'complete','out':str(out/'result.json'),'wall_seconds':result['wall_seconds']}),flush=True)

if __name__=='__main__':main()
