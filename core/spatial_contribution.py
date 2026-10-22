"""Scale-invariant static spatial-dependence diagnostic for arbitrary expressions.

Inputs are the fitted equation, field names, observed field/derivative values,
weak projection and source values. There is no reference equation or support.
This module measures resolved spatial contribution, not correctness of a PDE.
"""
from dataclasses import dataclass

import numpy as np
import sympy as sp


@dataclass(frozen=True)
class ContributionPolicy:
    relative_floor: float = 1e-6
    roundoff_multiplier: float = 64.0


def normalize_and_isolate(expressions, coefficients, algebra, states):
    state_functions = [sp.Function(state)(*algebra.coords) for state in states]
    terms = [sp.Float(str(float(coefficient)), 30) * algebra.normalized(expression)
             for expression, coefficient in zip(expressions, coefficients)]
    combined = sp.expand(sum(terms))
    derivatives = sorted((d for d in combined.atoms(sp.Derivative)
                          if any(d.expr.has(state) for state in state_functions)), key=str)
    if not derivatives:
        return combined, sp.Integer(0), [], 'exact_algebraic_relation'
    zero_derivatives = {d: sp.Integer(0) for d in derivatives}
    baseline = combined.xreplace(zero_derivatives)
    if baseline.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
        # Some legitimate laws contain reciprocal derivatives. In that case,
        # use a local differential response without replacing denominators by zero.
        jets = sp.symbols('q:' + str(len(derivatives)))
        substitution = dict(zip(derivatives, jets))
        value = combined.xreplace(substitution)
        spatial = sp.expand(sum(jet * sp.diff(value, jet) for jet in jets)).xreplace(dict(zip(jets, derivatives)))
        mode = 'local_derivative_scale_response'
    else:
        spatial = sp.expand(combined - baseline)
        mode = 'zero_derivative_difference'
    return combined, spatial, terms, mode


def measure_contribution(expressions, coefficients, algebra, states,
                         evaluate_symbolic, weak_project, source,
                         policy=ContributionPolicy()):
    combined, spatial, raw_terms, mode = normalize_and_isolate(expressions, coefficients, algebra, states)
    source = np.asarray(source, dtype=float)
    weak_source = np.asarray(weak_project(source), dtype=float)
    point_scale, weak_scale = np.linalg.norm(source), np.linalg.norm(weak_source)
    if min(point_scale, weak_scale) <= 0:
        return {'status': 'zero_source_requires_external_scale', 'resolved': False,
                'formula': str(combined), 'spatial_formula': str(spatial)}
    values = np.asarray(evaluate_symbolic(spatial), dtype=float)
    if not np.all(np.isfinite(values)):
        return {'status': 'nonfinite_spatial_contribution', 'resolved': False,
                'formula': str(combined), 'spatial_formula': str(spatial)}
    weak_values = np.asarray(weak_project(values), dtype=float)
    point_ratio = float(np.linalg.norm(values) / point_scale)
    weak_ratio = float(np.linalg.norm(weak_values) / weak_scale)
    raw_norm_sum = 0.0
    for term in raw_terms:
        evaluated = np.asarray(evaluate_symbolic(term), dtype=float)
        if np.all(np.isfinite(evaluated)):
            raw_norm_sum += np.linalg.norm(weak_project(evaluated)) / weak_scale
    roundoff = policy.roundoff_multiplier * np.finfo(float).eps * raw_norm_sum
    threshold = max(policy.relative_floor, roundoff)
    resolved = min(point_ratio, weak_ratio) > threshold
    return {'status': 'resolved_spatial_contribution' if resolved else 'spatial_contribution_unresolved',
            'resolved': bool(resolved), 'formula': str(combined), 'spatial_formula': str(spatial),
            'mode': mode, 'pointwise_ratio': point_ratio, 'weak_ratio': weak_ratio,
            'threshold': threshold, 'roundoff_estimate': float(roundoff),
            'floor_sensitivity': {str(floor): bool(min(point_ratio, weak_ratio) > max(floor, roundoff))
                                  for floor in (1e-8, 1e-6, 1e-4, 1e-3, 1e-2)}}
