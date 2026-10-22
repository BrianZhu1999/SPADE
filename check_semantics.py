"""Analytic identities test differentiation, weak integration and dimensions."""
import json
import numpy as np
import torch
from core.expressions import Expr,Grammar,make,Algebra
from core.fields import Block,Evaluator

def main():
    device='cuda:0';torch.set_grad_enabled(False);checks=[]
    t=np.linspace(0,1,41);x=np.arange(128)*2*np.pi/128;y=np.arange(48)*2*np.pi/48
    u=np.exp(t[:,None])*np.sin(x[None,:])
    b=Block({'u':u,'1':np.ones_like(u)},{'t':t,'x':x},['x'],{'x':'periodic'},device,'train')
    e=Evaluator(b);leaf=Expr('u')
    for order in range(1,5):
        expected=np.exp(t[:,None])*np.sin(x[None,:]+order*np.pi/2)
        derivative=b.derivative(e.field(leaf),'x'*order).cpu().numpy().reshape(u.shape)
        error=np.linalg.norm(expected-derivative)/np.linalg.norm(expected)
        assert error<1e-7,(order,error)
        weak=e.weak([Expr('D'+'x'*order,(leaf,))]).cpu().numpy()
        reference=b.project(torch.as_tensor(expected.reshape(-1,1),device=device)).cpu().numpy()
        werror=np.linalg.norm(weak-reference)/np.linalg.norm(reference)
        assert werror<1e-5,(order,werror)
        checks.append({'order':order,'strong_relative_error':error,'weak_relative_error':werror})
    v=np.cos(t[:,None,None])*np.sin(x[None,:,None])*np.cos(2*y[None,None,:])
    b2=Block({'u':v,'1':np.ones_like(v)},{'t':t,'x':x,'y':y},['x','y'],{'x':'periodic','y':'periodic'},device,'train')
    ev=Evaluator(b2)
    got=b2.project(ev.field(leaf),time_order=2)
    expected=-b2.project(ev.field(leaf))
    error=float(torch.linalg.norm(got-expected)/torch.linalg.norm(expected))
    assert error<1e-5,error
    g=Grammar({'rho':(1,-3,0),'u':(0,1,-1)},['x'])
    assert not g.allowed(make('Add',Expr('rho'),Expr('u')))
    assert not g.allowed(make('Sin',Expr('u')))
    assert g.allowed(make('Mul',Expr('rho'),Expr('u')))
    assert g.allowed(make('Div',Expr('rho'),Expr('u')))
    assert make('Div',leaf,Expr('1')).order==0
    expr=make('Dx',make('Mul',leaf,leaf));alt=make('Mul',leaf,make('Dx',leaf))
    alg=Algebra([expr,alt]);assert alg.key(0)==alg.key(1)
    h=Expr('h');one=Expr('1')
    constant=make('Div',make('Add',h,one),make('Add',one,h))
    ratio=make('Mul',make('Add',make('Add',one,one),make('Add',h,h)),make('Div',make('Sin',one),make('Add',one,h)))
    alg2=Algebra([constant,one,ratio]);assert alg2.key(0)==alg2.key(1)==alg2.key(2)
    ev.weak([make('Dx',make('Sin',leaf)),make('Dxx',make('Sin',leaf))])
    assert make('Sin',leaf) in ev.cache
    fresh=Evaluator(b)
    square=make('Mul',leaf,leaf);cube=make('Mul',square,leaf)
    fresh.batch([cube,square])
    assert fresh.computed==len(fresh.cache)==3,(fresh.computed,len(fresh.cache))
    fresh.batch([square,cube])
    assert fresh.computed==3
    exprs=[make('Dxx',make('Add',square,make('Sin',leaf))),make('Add',square,cube)]
    cached=fresh.weak(exprs);direct=fresh.weak_direct(exprs)
    assert float(torch.max(torch.abs(cached-direct)))<1e-10
    sampled=Block({'u':v,'1':np.ones_like(v)},{'t':t,'x':x,'y':y},['x','y'],{'x':'periodic','y':'periodic'},device,'validation',weak_rows=128)
    sparse_values=sampled.project(sampled.fields['u'])
    full_values=b2.project(b2.fields['u'])[sampled.row_indices]
    assert torch.allclose(sparse_values,full_values,rtol=1e-12,atol=1e-12)
    print(json.dumps({'checks':'passed','one_dimensional':checks,'two_dimensional_second_time_error':error},indent=2))

if __name__=='__main__':main()
