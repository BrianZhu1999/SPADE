"""Symbolic conservation diagnostics for scalar spatial PDEs.

The audit is deliberately post hoc: it checks a selected right-hand side
without changing the search space or the model-selection rule.  A scalar
one-dimensional expression is a local flux derivative when its Euler
operator with respect to the state vanishes.
"""
from __future__ import annotations

import sympy as sp

from .expressions import Algebra, Expr, is_diff


def _state_function(name: str, coords):
    return sp.Function(str(name))(*coords)


def _jet_derivatives(value, state, axis):
    found = [state]
    for atom in value.atoms(sp.Derivative):
        if atom.expr != state:
            continue
        variables = tuple(atom.variable_count)
        if all(symbol == axis for symbol, _ in variables):
            found.append(atom)
    unique = list(dict.fromkeys(found))
    return sorted(unique, key=lambda item: 0 if item == state else len(item.variable_count))


def _axis_symbol(axis: str):
    if axis not in "xyz":
        raise ValueError(f"axis must be one of x, y, z; got {axis!r}")
    return sp.symbols("x y z")["xyz".index(axis)]


def _coefficient(value):
    """Use the decimal representation as an exact symbolic coefficient."""
    if isinstance(value, sp.Basic):
        return value
    return sp.Rational(str(value))


def euler_operator(value, state_name: str, axis: str = "x"):
    """Compute the one-dimensional variational derivative with respect to a state."""
    axis_symbol = _axis_symbol(axis)
    coords = sp.symbols("x y z")
    state = _state_function(state_name, coords)
    result = sp.Integer(0)
    for jet in _jet_derivatives(value, state, axis_symbol):
        order = 0 if jet == state else sum(count for _, count in jet.variable_count)
        partial = sp.diff(value, jet)
        result += (-1) ** order * sp.diff(partial, axis_symbol, order)
    return sp.simplify(sp.expand(result))


def _strip_outer_axis_derivative(expression: Expr, axis: str):
    """Return a flux AST when the expression has an explicit outer ``D_axis``.

    This is a conservative reconstruction.  It never claims that a general
    Euler-zero expression has been integrated; such combinations remain
    auditable but return ``None`` here.
    """
    if expression.op == "D" + axis:
        return expression.args[0]
    if is_diff(expression.op) and expression.op.startswith("D"):
        axes = expression.op[1:]
        if axes.startswith(axis):
            remainder = axes[1:]
            return expression.args[0] if not remainder else Expr("D" + remainder, expression.args)
    return None


def explicit_flux(expression: Expr, axis: str = "x"):
    """Return a conservative flux AST for an explicitly outer-differentiated term."""
    if axis not in "xyz":
        raise ValueError(f"axis must be one of x, y, z; got {axis!r}")
    return _strip_outer_axis_derivative(expression, axis)


def audit_expression(expression: Expr, coefficient: float = 1.0,
                     state_name: str = "u", axis: str = "x"):
    algebra = Algebra([])
    weighted = _coefficient(coefficient) * algebra.symbolic(expression)
    residual = euler_operator(weighted, state_name, axis)
    flux = explicit_flux(expression, axis)
    return {
        "expression": expression.text(),
        "coefficient": float(coefficient),
        "euler_operator": sp.sstr(residual),
        "is_total_spatial_derivative": bool(residual == 0),
        "explicit_flux": flux.text() if flux is not None else None,
    }


def audit_terms(terms, state_name: str = "u", axis: str = "x"):
    algebra = Algebra([])
    combined = sp.Integer(0)
    rows = []
    for expression, coefficient in terms:
        value = algebra.symbolic(expression)
        combined += _coefficient(coefficient) * value
        rows.append(audit_expression(expression, coefficient, state_name, axis))
    residual = euler_operator(sp.expand(combined), state_name, axis)
    explicit_fluxes = [explicit_flux(expression, axis) for expression, _ in terms]
    return {
        "criterion": "Euler operator vanishes for a local one-dimensional flux derivative",
        "state": state_name,
        "axis": axis,
        "equation_euler_operator": sp.sstr(residual),
        "equation_euler_operator_numeric": sp.sstr(sp.N(residual, 16)),
        "equation_is_total_spatial_derivative": bool(residual == 0),
        "explicit_flux": ([
            ("+" if i == 0 else "") + f"({coefficient})*{flux.text()}"
            for i, ((_, coefficient), flux) in enumerate(zip(terms, explicit_fluxes))
            if flux is not None
        ] if all(flux is not None for flux in explicit_fluxes) else None),
        "terms": rows,
    }


def is_flux_expression(expression: Expr, axis: str | None = None):
    """Return True when an AST has an outer spatial derivative.

    With ``axis`` supplied, the outer derivative must start in that direction.
    The default keeps the historical multi-direction behavior used by the
    three-dimensional flux case.
    """
    if not is_diff(expression.op):
        return False
    if axis is None:
        return any(direction in "xyz" for direction in expression.op[1:])
    if axis not in "xyz":
        raise ValueError(f"axis must be one of x, y, z; got {axis!r}")
    return expression.op.startswith("D" + axis)

