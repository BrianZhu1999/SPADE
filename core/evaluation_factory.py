"""Observation-grid selection of generic evaluators and static-dependence checks."""
from collections import OrderedDict
import numpy as np
import sympy as sp
import torch

from .conservative_evaluator import ConservativeAlgebra, ConservativeEvaluator
from .expressions import Algebra
from .fields import Evaluator
from .spatial_contribution import ContributionPolicy, measure_contribution


def discretization_settings(metadata):
    setting = metadata.get('discretization', {})
    return setting if isinstance(setting, dict) else {}


def create_evaluator(block, metadata=None):
    metadata = metadata if metadata is not None else getattr(block, 'observation_metadata', {})
    setting = discretization_settings(metadata)
    if setting.get('type') == 'conservative_faces':
        return ConservativeEvaluator(block, boundary_extensions=setting.get('boundary_extensions'))
    return Evaluator(block)


def algebra_class(metadata):
    return ConservativeAlgebra if discretization_settings(metadata).get('type') == 'conservative_faces' else Algebra


def create_algebra(expressions, metadata):
    return algebra_class(metadata)(expressions)


class SpatialContributionGuard:
    """Callable on complete equations, with training fields captured at creation.

    API: guard(expressions, coefficients) -> bool. Details for the most recently
    checked equation are retained in last_report. The relative floor is a method
    configuration, and does not come from reference equations or observation metadata.
    """
    def __init__(self, metadata, evaluators, target_index=0, relative_floor=1e-6):
        self.metadata = metadata
        self.states = [f['name'] for f in metadata['fields'] if f.get('role') == 'state']
        target = metadata['targets'][target_index]
        if target.get('unknown_field'):
            self.states = [target['unknown_field']]
        self.source_name = target['field']
        self.time_order = int(target.get('time_order', 1))
        self.blocks = [ev.block for ev in evaluators if ev.block.split == 'train']
        self.policy = ContributionPolicy(relative_floor=relative_floor)
        self.algebra = create_algebra([], metadata)
        self.atom_cache = [{} for _ in self.blocks]
        self.last_report = None
        self.checked = 0
        self.rejected = 0
        self.cache_hits = 0
        self.result_cache = OrderedDict()
        self.max_cache_entries = 1024

    def _numeric(self, value, block, cache):
        if value.is_Number:
            return torch.full((int(np.prod(block.shape)), 1), float(value), device=block.device, dtype=torch.float64)
        if value in cache:
            return cache[value]
        if isinstance(value, sp.Derivative):
            child = self._numeric(value.expr, block, cache)
            letters = ''.join(str(axis)*count for axis, count in value.variable_count)
            answer = block.derivative(child, letters)
            if value.expr.func.__name__ in block.fields:
                cache[value] = answer
            return answer
        if value.func.__name__ in block.fields:
            return block.fields[value.func.__name__]
        args = [self._numeric(child, block, cache) for child in value.args]
        if value.func == sp.Add:
            return sum(args)
        if value.func == sp.Mul:
            result = args[0]
            for arg in args[1:]: result = result*arg
            return result
        if value.func == sp.Pow: return args[0]**args[1]
        return {sp.sin:torch.sin, sp.cos:torch.cos, sp.tanh:torch.tanh}[value.func](args[0])

    def __call__(self, expressions, coefficients):
        if self.time_order != 0:
            return True
        key = (tuple(expressions), tuple(float(c) for c in coefficients))
        if key in self.result_cache:
            self.cache_hits += 1
            admissible, self.last_report = self.result_cache[key]
            self.result_cache.move_to_end(key)
            return admissible
        reports = []
        for block, cache in zip(self.blocks, self.atom_cache):
            source = block.fields[self.source_name].detach().cpu().numpy().ravel()
            def evaluate(value):
                return self._numeric(value, block, cache).detach().cpu().numpy().ravel()
            def project(values):
                data = torch.as_tensor(np.asarray(values).reshape(-1,1),device=block.device,dtype=torch.float64)
                return block.project(data).detach().cpu().numpy().ravel()
            reports.append(measure_contribution(expressions, coefficients, self.algebra,
                                               self.states, evaluate, project, source, self.policy))
        admissible = bool(reports) and all(row['resolved'] for row in reports)
        self.checked += 1
        self.rejected += int(not admissible)
        self.last_report = {'training_only':True, 'admissible':admissible, 'blocks':reports,
                            'relative_floor':self.policy.relative_floor,
                            'numerical_role':'state-spatial amplitude check using nodal derivatives and weak projection; not a conservative discrete equation residual'}
        self.result_cache[key] = (admissible, self.last_report)
        if len(self.result_cache) > self.max_cache_entries:
            self.result_cache.popitem(last=False)
        return admissible


def create_spatial_guard(metadata, evaluators, target_index=0, relative_floor=1e-6):
    return SpatialContributionGuard(metadata, evaluators, target_index, relative_floor)
