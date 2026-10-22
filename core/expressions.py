"""Generic typed expression trees. No equation-specific production rules."""
from dataclasses import dataclass
from functools import cached_property
import itertools
import sympy as sp

def is_diff(op):return len(op)>1 and op[0]=='D' and set(op[1:])<=set('xyz')

@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple = ()
    @cached_property
    def size(self): return 1 + sum(a.size for a in self.args)
    @cached_property
    def order(self): return max((a.order for a in self.args), default=0) + (len(self.op)-1 if is_diff(self.op) else 0)
    @cached_property
    def depth(self): return 0 if not self.args else 1 + max(a.depth for a in self.args)
    @cached_property
    def operators(self):return frozenset([self.op]).union(*(a.operators for a in self.args))
    def text(self): return self.op if not self.args else self.op+'('+','.join(a.text() for a in self.args)+')'
    def pack(self): return [self.op, *[a.pack() for a in self.args]]

def unpack(data): return Expr(data[0], tuple(unpack(a) for a in data[1:]))

def make(op, a, b=None):
    if op == 'Identity': return a
    if op in ('Add','Mul'):
        a,b = sorted((a,b), key=lambda e:e.text())
        if op == 'Mul' and a.op == '1': return b
    if op == 'Div' and b.op == '1': return a
    if is_diff(op) and is_diff(a.op):
        return Expr('D'+''.join(sorted(op[1:]+a.op[1:])), a.args)
    return Expr(op, (a,) if b is None else (a,b))

class Grammar:
    def __init__(self, units, axes, coord_units=None, dimensions=True, max_order=4, max_nodes=17,
                 operator_prior='open'):
        self.units = {k:tuple(v) for k,v in units.items()}
        self.units['1'] = (0,0,0)
        self.axes = tuple(axes)
        self.coords = coord_units or {a:(0,1,0) for a in axes}
        self.dimensions,self.max_order,self.max_nodes = dimensions,max_order,max_nodes
        # A shared polynomial differential prior for difficult fluid cases.
        # It is independent of the target equation: products, sums and spatial
        # derivatives up to the requested order are retained; transcendental
        # and rational operators are excluded.
        self.operator_prior = operator_prior
        self.cache = {}
        self.counts = {'generated':0,'dimension_rejected':0,'complexity_rejected':0,'duplicate':0}
    def dim(self, e):
        if e in self.cache:return self.cache[e]
        if not e.args:d=self.units[e.op]
        else:
            a=self.dim(e.args[0]);b=self.dim(e.args[1]) if len(e.args)==2 else None
            if a is None or (len(e.args)==2 and b is None):d=None
            elif e.op=='Add':d=a if a==b or not self.dimensions else None
            elif e.op=='Mul':d=tuple(x+y for x,y in zip(a,b))
            elif e.op=='Div':d=tuple(x-y for x,y in zip(a,b))
            elif is_diff(e.op):
                d=tuple(a[i]-sum(self.coords[q][i] for q in e.op[1:]) for i in range(3))
            else:d=(0,0,0) if a==(0,0,0) or not self.dimensions else None
        self.cache[e]=d;return d
    def allowed(self,e):
        self.counts['generated']+=1
        if e.order>self.max_order or e.size>self.max_nodes:
            self.counts['complexity_rejected']+=1;return False
        if self.dim(e) is None:self.counts['dimension_rejected']+=1;return False
        if is_diff(e.op) and e.args[0].op=='1':return False
        if e.op=='Div' and e.args[0]==e.args[1]:return False
        return True
    def operations(self):
        if self.operator_prior in ('polynomial_differential', 'flux_conservation'):
            # Monomial flux factors avoid hiding several independent
            # mechanisms inside one additive candidate column.
            ops=['Mul']
        else:
            ops=['Add','Mul','Div','Sin','Cos','Tanh']
        return ops+['D'+a*k for a in self.axes for k in range(1,self.max_order+1)]
    def stream(self,pool,op,seen):
        pairs = itertools.product(pool,repeat=2) if op=='Div' else itertools.combinations_with_replacement(pool,2)
        iterator=(make(op,a,b) for a,b in pairs) if op in ('Add','Mul','Div') else (make(op,a) for a in pool)
        for e in iterator:
            if not self.allowed(e):continue
            if e in seen:self.counts['duplicate']+=1;continue
            seen.add(e);yield e

class Algebra:
    """Exact multivariate differential equivalence, independent of observations."""
    def __init__(self,exprs):
        self.exprs=exprs;self.coords=sp.symbols('x y z');self.trees={};self.keys={}
    def symbolic(self,e):
        if e in self.trees:return self.trees[e]
        if e.op=='1':v=sp.Integer(1)
        elif not e.args:v=sp.Function(e.op)(*self.coords)
        else:
            args=[self.symbolic(a) for a in e.args]
            if e.op=='Add':v=args[0]+args[1]
            elif e.op=='Mul':v=args[0]*args[1]
            elif e.op=='Div':v=args[0]/args[1]
            elif is_diff(e.op):
                v=args[0]
                for axis in e.op[1:]:v=sp.diff(v,self.coords['xyz'.index(axis)])
            else:v={'Sin':sp.sin,'Cos':sp.cos,'Tanh':sp.tanh}[e.op](args[0])
        self.trees[e]=v;return v
    def key(self,index):
        if index in self.keys:return self.keys[index]
        value=self.normalized(self.exprs[index])
        if value==0:key='ZERO'
        else:
            first=sp.Add.make_args(value)[0].as_independent(*self.coords,as_Add=False)[0]
            key=sp.srepr(sp.expand(value/first))
        self.keys[index]=key;return key

    def normalized(self,e):
        value=sp.expand(self.symbolic(e))
        if 'Div' in e.operators:value=sp.expand(sp.cancel(value))
        return value
