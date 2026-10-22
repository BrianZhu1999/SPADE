"""Training-only conditional-gain search with exact expression diversity.

All columns come from the recursive algebraic-differential grammar.
Validation chooses only after training search and backward deletion finish.
"""
from dataclasses import dataclass
import itertools
import json
import time

import numpy as np
import sympy as sp
import torch

CAPS = (2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64)
BEAM_WIDTH = 64
PER_BUDGET = 4
TOP_GLOBAL = 128
TOP_PER_SIZE = 32
PENALTY = 1e-8
PRUNE = 1e-5
SCAN = 8192


def stable_lstsq(matrix, target, rcond=1e-10, method="svd"):
    """Least-squares solve with an explicit QR or SVD factorization."""
    if method == "qr":
        q, r = np.linalg.qr(matrix, mode='reduced')
        singular_values = np.linalg.svd(r, compute_uv=False)
        cutoff = rcond * (singular_values[0] if singular_values.size else 0.0)
        rank = int(np.count_nonzero(singular_values > cutoff))
        if rank == matrix.shape[1]:
            coef = np.linalg.solve(r, q.T @ target)
        else:
            coef = np.linalg.pinv(matrix, rcond=rcond) @ target
    elif method == "svd":
        u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
        cutoff = rcond * (singular_values[0] if singular_values.size else 0.0)
        keep = singular_values > cutoff
        rank = int(np.count_nonzero(keep))
        coef = vh[keep].T @ ((u[:, keep].T @ target) / singular_values[keep])
    else:
        raise ValueError(f"unknown fit method: {method}")
    return coef, rank, singular_values


def robust_lstsq(matrix, target, rcond=1e-10, method="qr", huber_delta=1.5,
                 iterations=6):
    """Huber iteratively reweighted least squares on training rows only."""
    coef, rank, singular_values = stable_lstsq(matrix, target, rcond=rcond, method=method)
    for _ in range(max(1, int(iterations))):
        residual = target - matrix @ coef
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        scale = max(float(scale), 1e-12)
        z = np.abs(residual) / (float(huber_delta) * scale)
        weights = np.minimum(1.0, 1.0 / np.maximum(z, 1.0))
        root = np.sqrt(weights)
        coef, rank, singular_values = stable_lstsq(matrix * root[:, None], target * root,
                                                    rcond=rcond, method=method)
    return coef, rank, singular_values


@dataclass
class State:
    support: tuple
    coef: np.ndarray
    mse: float
    complexity: int


class Algebra:
    """Only exact algebraic identities remove duplicate scalar columns."""
    def __init__(self, exprs):
        self.exprs = exprs
        self.x = sp.Symbol('x')
        self.u = sp.Function('u')(self.x)
        self.jets = sp.symbols('u q1 q2 q3 q4 q5 q6 q7 q8 q9')
        self.trees = {}
        self.keys = {}

    def symbolic(self, e):
        key = e.text()
        if key in self.trees:
            return self.trees[key]
        if e.op == '1': v = sp.Integer(1)
        elif e.op == 'u': v = self.u
        else:
            a = [self.symbolic(c) for c in e.args]
            if e.op == 'Add': v = a[0] + a[1]
            elif e.op == 'Mul': v = a[0] * a[1]
            elif e.op == 'Div': v = a[0] / a[1]
            elif e.op == 'Sin': v = sp.sin(a[0])
            elif e.op == 'Cos': v = sp.cos(a[0])
            elif e.op == 'Tanh': v = sp.tanh(a[0])
            elif e.op.startswith('D'): v = sp.diff(a[0], self.x, len(e.op)-1)
            else: raise ValueError(e.op)
        self.trees[key] = v
        return v

    def key(self, index):
        if index not in self.keys:
            raw = self.symbolic(self.exprs[index])
            sub = {sp.diff(self.u, self.x, k): self.jets[k] for k in range(1, 10)}
            sub[self.u] = self.jets[0]
            value = sp.expand(raw.xreplace(sub))
            # Coefficients are fitted independently; overall nonzero constants
            # do not create distinct columns. No data-similarity collapse.
            independent, dependent = value.as_independent(*self.jets, as_Add=False)
            if independent.is_zero:
                key = 'ZERO'
            else:
                terms = sp.Add.make_args(dependent)
                first = terms[0].as_independent(*self.jets, as_Add=False)[0]
                key = sp.srepr(sp.expand(dependent / first)) if first != 0 else sp.srepr(dependent)
            self.keys[index] = key
        return self.keys[index]


def select_equations(table, target, exprs, ntrain_rows, device, max_terms=8, deadline=None,
                     support_is_independent=None, max_condition=None,
                     fit_method="svd",final_admissible=None,complexity_bounds=None,
                     swap_rounds=2, swap_beam=16, robust_fit=False,
                     huber_delta=1.5, irls_iterations=6):
    started = time.perf_counter()
    raw = table.cpu().double().numpy()
    yy = target.cpu().double().numpy()
    scales = np.sqrt(np.einsum('ij,ij->j',raw[:ntrain_rows],raw[:ntrain_rows])/ntrain_rows)
    valid = np.isfinite(scales) & (scales > 1e-12)
    scales[~valid] = 1
    z = raw / scales
    z[:, ~valid] = 0
    train = z[:ntrain_rows]
    sizes = np.array([e.size for e in exprs], dtype=np.int32)
    algebra = Algebra(exprs)
    # Close columns require double precision before ranking, not merely after
    # screening. Tiny projected norms are recomputed by explicit subtraction.
    gpu = torch.as_tensor(train, dtype=torch.float64, device=device)
    norms = torch.linalg.vector_norm(gpu,dim=0).square()
    gpu_sizes = torch.as_tensor(sizes, device=device)
    selected, paths = [], []
    for eq in range(yy.shape[1]):
        check_admissible=final_admissible[eq] if isinstance(final_admissible,(list,tuple)) else final_admissible
        # All equations share one wall-clock deadline supplied by the outer
        # search.  Per-equation reserve windows caused unfinished searches to
        # stop early while other equations still had available budget.
        eq_deadline = deadline
        y = yy[:ntrain_rows, eq]
        yrms = max(float(np.sqrt(np.mean(y*y))), 1e-12)
        cache = {}
        catalogue = {}

        def fit(indices):
            support = tuple(sorted(set(indices)))
            if len(support) > max_terms:
                raise ValueError('Search support exceeds the configured term budget')
            if support in cache: return cache[support]
            if not support:
                state = State((), np.empty(0), 1., 0)
            else:
                if complexity_bounds is not None and complexity_bounds[eq] is not None and int(sizes[list(support)].sum())>=complexity_bounds[eq]:
                    cache[support]=None
                    return None
                if support_is_independent is not None and not support_is_independent(support):
                    cache[support] = None
                    return None
                matrix = train[:, support]
                if robust_fit:
                    coef, rank, singular_values = robust_lstsq(
                        matrix, y, rcond=1e-10, method=fit_method,
                        huber_delta=huber_delta, iterations=irls_iterations)
                else:
                    coef, rank, singular_values = stable_lstsq(
                        matrix, y, rcond=1e-10, method=fit_method)
                if rank != len(support) or (max_condition is not None and
                       singular_values[0] / singular_values[-1] > max_condition):
                    cache[support] = None
                    return None
                residual = y - matrix @ coef
                state = State(support, coef, float(np.mean(residual**2)/yrms**2), int(sizes[list(support)].sum()))
                if check_admissible is not None and state.mse<1e-10 and not check_admissible(support,coef/scales[list(support)]):
                    cache[support]=None
                    return None
            cache[support] = state
            if support: catalogue[support] = state
            return state

        def signature(state):
            return tuple(sorted(algebra.key(i) for i in state.support))

        def retain(states):
            states = sorted({s.support: s for s in states if s is not None}.values(),
                            key=lambda s: (s.complexity, s.mse, s.support))
            distinct = {}
            for state in states:
                key = signature(state)
                if key not in distinct: distinct[key] = state
            pool = list(distinct.values())
            keep = {}
            for cap in CAPS:
                eligible = sorted((s for s in pool if s.complexity <= cap), key=lambda s: (s.mse, s.complexity))
                for state in eligible[:PER_BUDGET]: keep[state.support] = state
            for state in sorted(pool, key=lambda s: (s.mse+1e-9*s.complexity, s.complexity)):
                if len(keep) >= BEAM_WIDTH: break
                keep.setdefault(state.support, state)
            return list(keep.values())[:BEAM_WIDTH]

        def additions(beam, include_elementary=False):
            """Rank additions using training rows only, for forward or swap steps."""
            # Only active models need observation-length residual arrays.
            # Storing one per fitted support grows as fits * observations.
            residuals = torch.as_tensor(np.column_stack([y-train[:,s.support]@s.coef if s.support else y for s in beam]), dtype=torch.float64, device=device)
            qparts = []
            offsets = [0]
            for state in beam:
                q = np.linalg.qr(train[:, state.support], mode='reduced')[0] if state.support else np.empty((len(y), 0))
                qparts.append(q)
                offsets.append(offsets[-1] + q.shape[1])
            qgpu = torch.as_tensor(np.concatenate(qparts, axis=1), dtype=torch.float64, device=device)
            # Accumulate per-state score vectors to make conditional ranking
            # over ALL columns, before any correlation shortlist is applied.
            scores = torch.empty((gpu.shape[1], len(beam)), dtype=torch.float64, device=device)
            for begin in range(0, gpu.shape[1], SCAN):
                end = min(begin+SCAN, gpu.shape[1])
                block = gpu[:, begin:end]
                dots = block.T @ residuals
                proj = qgpu.T @ block
                for j, state in enumerate(beam):
                    den = norms[begin:end] - proj[offsets[j]:offsets[j+1]].square().sum(0)
                    close = den <= len(y)*1e-6
                    if state.support and close.any():
                        qj = qgpu[:,offsets[j]:offsets[j+1]]
                        orthogonal_remainder = block[:,close] - qj @ proj[offsets[j]:offsets[j+1],close]
                        den[close] = orthogonal_remainder.square().sum(0)
                    score = dots[:,j].square() / den.clamp_min(1e-20)
                    score[den <= len(y)*1e-20] = -torch.inf
                    scores[begin:end,j] = score
            candidates = []
            for j, state in enumerate(beam):
                shortlist = set(torch.topk(scores[:,j], min(TOP_GLOBAL,len(exprs))).indices.cpu().tolist())
                for low, high in zip((0,*CAPS[:-1]), CAPS):
                    inds = torch.nonzero((gpu_sizes > low)&(gpu_sizes <= high), as_tuple=True)[0]
                    if len(inds):
                        order = torch.topk(scores[inds,j], min(TOP_PER_SIZE,len(inds))).indices
                        shortlist.update(inds[order].cpu().tolist())
                if include_elementary: shortlist.update(elementary.values())
                # Correct shortlist scores in double precision, then collapse
                # only exact symbolic duplicates; retain simple representatives.
                support_keys = {algebra.key(i) for i in state.support}
                options = {}
                for index in sorted(shortlist, key=lambda i:(sizes[i],i)):
                    key = algebra.key(index)
                    if key in support_keys or key == 'ZERO': continue
                    options.setdefault(key,index)
                for index in options.values():
                    candidate = fit((*state.support,index))
                    if candidate is not None: candidates.append(candidate)
            return candidates

        beam = [fit(())]
        # Every elementary unary expression starts a branch. A mechanism whose
        # individual correlation is small can still be essential jointly.
        elementary = {}
        for i in np.flatnonzero(sizes <= 2):
            elementary.setdefault(algebra.key(int(i)), int(i))
        for step in range(1, max_terms+1):
            if eq_deadline is not None and time.perf_counter() > eq_deadline:
                print(json.dumps({'search_stopped':'budget_reserve_for_validation','step':step}),flush=True)
                break
            candidates = additions(beam, include_elementary=step == 1)
            if not candidates: break
            beam = retain(candidates)
            if step == 1:
                for index in elementary.values():
                    state = fit((index,))
                    if state is not None and state.support not in {s.support for s in beam}: beam.append(state)
            print(json.dumps({'selector':'conditional_diverse_beam','step':step,'beam':len(beam),
                              'fits':len(cache),'best_training_mse':min(s.mse for s in beam),
                              'seconds':time.perf_counter()-started}),flush=True)
            if any(s.mse<1e-10 and (check_admissible is None or check_admissible(s.support,s.coef/scales[list(s.support)])) for s in beam):
                print(json.dumps({'search_stopped':'training_numerical_tolerance','step':step}),flush=True)
                break
        # The term cap limits model size, not the number of opportunities to
        # correct an earlier choice. Delete before adding: even intermediate
        # fits respect max_terms. Reference equations and validation rows play
        # no role in proposing these exchanges.
        for round_index in range(swap_rounds):
            if min(s.mse for s in beam) < 1e-10:
                break
            if eq_deadline is not None and time.perf_counter() > eq_deadline:
                break
            parents = sorted(beam, key=lambda s:(s.mse, s.complexity, s.support))[:swap_beam]
            deletions = {}
            for state in parents:
                for dropped in state.support:
                    child = fit(i for i in state.support if i != dropped)
                    if child is not None:
                        deletions[child.support] = child
            starts = list(deletions.values())
            exchanged = []
            before = min(s.mse for s in beam)
            for begin in range(0, len(starts), BEAM_WIDTH):
                if eq_deadline is not None and time.perf_counter() > eq_deadline:
                    break
                exchanged.extend(additions(starts[begin:begin+BEAM_WIDTH]))
            beam = retain([*beam, *exchanged])
            after = min(s.mse for s in beam)
            print(json.dumps({'selector':'training_term_exchange','round':round_index+1,
                'parents':len(parents),'deletion_starts':len(starts),'fits':len(cache),
                'best_training_mse':after,'previous_training_mse':before,
                'max_intermediate_terms':max((len(s) for s in cache),default=0),
                'seconds':time.perf_counter()-started}),flush=True)
            if after >= before*(1-1e-10):
                break
        # Examine every deletion subset of promising supports. Previously a
        # mistaken early inclusion was irreversible unless its coefficient tiny.
        finalists = retain(list(catalogue.values()))
        for state in finalists:
            if eq_deadline is not None and time.perf_counter() > eq_deadline: break
            # At high term caps, test one/two deletions then iterate sparse
            # coefficient pruning; exhaustive 2**K subsets would be unbounded.
            counts=range(1,len(state.support)) if len(state.support)<=10 else (len(state.support)-1,len(state.support)-2)
            for count in counts:
                for subset in itertools.combinations(state.support,count): fit(subset)
        entries = []
        vy = yy[ntrain_rows:,eq]
        variance = max(float(np.mean(vy*vy)),1e-24)
        cleaned = {}
        for state in list(catalogue.values()):
            current = state
            for _ in range(max_terms):
                support = tuple(i for i,c in zip(current.support,current.coef) if abs(c)>=PRUNE*yrms)
                if support == current.support: break
                current = fit(support)
                if current is None: break
            if current is not None and current.support: cleaned[current.support] = current
        for state in cleaned.values():
            pred = z[ntrain_rows:,state.support] @ state.coef
            if not np.isfinite(pred).all(): continue
            vmse = float(np.mean((pred-vy)**2)/variance)
            entries.append({'score':vmse+PENALTY*state.complexity,'validation_relative_mse':vmse,
                            'training_relative_mse':state.mse,'complexity':state.complexity,
                            'terms':[{'coefficient':float(c/scales[i]),'expression':exprs[i].pack(),'text':exprs[i].text()}
                                     for i,c in zip(state.support,state.coef)],
                            'selector':'conditional_diverse_beam','validation_complexity_penalty':PENALTY})
        if not entries:
            if complexity_bounds is not None and complexity_bounds[eq] is not None:
                selected.append([]);paths.append([]);continue
            raise RuntimeError('No finite validation candidate')
        entries.sort(key=lambda e:(e['score'],e['complexity']))
        entries[0]['selected'] = True
        selected.append(entries[0]['terms'])
        paths.append(entries)
        print(json.dumps({'equation':eq,'selected':entries[0],'fits':len(cache),'symbolic_keys':len(algebra.keys)},ensure_ascii=False),flush=True)
    return selected, paths
