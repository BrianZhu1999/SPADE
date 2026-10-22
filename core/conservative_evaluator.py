"""Generic AST evaluator with arithmetic faces and exact weak summation by parts.

This evaluator is independent of case names, reference equations,
coefficient values and target supports. All operators use the same recursive
rules. It retains the public interface of core.fields.Evaluator.
"""
import sympy as sp
import torch

from .expressions import Algebra, is_diff, make
from .fields import Evaluator


class ConservativeAlgebra(Algebra):
    """Algebraic normalization with differential products kept unexpanded.

    For discontinuous coefficient fields, compact flux derivatives and expanded
    pointwise derivative products have different discrete meanings. Differentiation
    is therefore a primitive symbolic object; normal algebra still cancels and
    combines identical expressions and coefficients.
    """
    def symbolic(self, expression):
        if expression in self.trees:
            return self.trees[expression]
        if is_diff(expression.op):
            value = self._symbolic_derivative(
                self.symbolic(expression.args[0]), expression.op[1:])
            self.trees[expression] = value
            return value
        return super().symbolic(expression)

    def _symbolic_derivative(self, value, letters):
        """Cancel derivatives of constants and distribute only over sums.

        Product derivatives remain unevaluated so conservative flux terms keep
        their compact discrete meaning.  This prevents a static equation from
        passing the spatial-dependence guard through a cancelling identity such
        as ``Dyyyy(h) - Dyyyy(1+h) + 1``.
        """
        if value.is_Number:
            return sp.Integer(0)
        if value.is_Add:
            terms = [self._symbolic_derivative(term, letters) for term in value.args]
            return sp.Add(*terms)
        return sp.Derivative(
            value,
            *[self.coords['xyz'.index(axis)] for axis in letters],
            evaluate=False,
        )


class ConservativeEvaluator(Evaluator):
    def __init__(self, block, boundary_extensions=None):
        super().__init__(block)
        self.boundary_extensions = boundary_extensions
        self.face_cache = {}
        self.conservative_weak_cache = {}
        self.face_nodes_computed = 0
        self.face_cache_hits = 0

    def _shape(self, value):
        return value.reshape(*self.block.shape, -1)

    def _periodic(self, axis):
        return self.block.boundary.get(self.block.axes[axis]) == 'periodic'

    def _spacing(self, axis):
        coord = self.block.coords[self.block.axes[axis]]
        return float(coord[1] - coord[0])

    def _neighbors(self, value, axis):
        if self._periodic(axis):
            return value, torch.roll(value, -1, dims=axis)
        first = [slice(None)] * value.ndim
        second = first.copy()
        first[axis], second[axis] = slice(None, -1), slice(1, None)
        return value[tuple(first)], value[tuple(second)]

    def _interpolate(self, value, axis):
        left, right = self._neighbors(value, axis)
        return (left + right) / 2

    def _ghost_cell(self, expression, axis, side):
        if not expression.args:
            if expression.op == '1':
                rule = 'copy'
            else:
                rule = self.boundary_extensions[expression.op]
            cell = self._shape(self.field(expression)).narrow(axis, 0 if side == 0 else self.block.shape[axis]-1, 1)
            if rule == 'copy': return cell
            if rule == 'zero': return torch.zeros_like(cell)
            if isinstance(rule, (int, float)): return torch.full_like(cell, rule)
            raise ValueError(f'Unsupported ghost extension {rule}')
        if is_diff(expression.op):
            # Higher or tangential derivative ghost values use the edge
            # one-sided derivative already defined by the observation block.
            return self._shape(self.field(expression)).narrow(axis, 0 if side == 0 else self.block.shape[axis]-1, 1)
        args = [self._ghost_cell(child, axis, side) for child in expression.args]
        if expression.op == 'Add': return args[0]+args[1]
        if expression.op == 'Mul': return args[0]*args[1]
        if expression.op == 'Div': return args[0]/args[1]
        return {'Sin':torch.sin,'Cos':torch.cos,'Tanh':torch.tanh}[expression.op](args[0])

    def _expression_neighbors(self, expression, axis):
        value = self._shape(self.field(expression))
        if self._periodic(axis) or self.boundary_extensions is None:
            return self._neighbors(value, axis)
        padded = torch.cat((self._ghost_cell(expression, axis, 0), value,
                            self._ghost_cell(expression, axis, 1)), dim=axis)
        first = [slice(None)] * value.ndim
        second = first.copy()
        first[axis], second[axis] = slice(None, -1), slice(1, None)
        return padded[tuple(first)], padded[tuple(second)]

    def face(self, expression, axis):
        """Evaluate a subtree at faces normal to a spatial coordinate.

        Algebra acts on face operands. A derivative in the normal direction
        uses adjacent cell values. Remaining tangential or higher derivatives
        are computed on cells, then transferred to the chosen face location.
        """
        key = (expression, axis)
        if key in self.face_cache:
            self.face_cache_hits += 1
            return self.face_cache[key].to(self.block.device)
        if not expression.args:
            left, right = self._expression_neighbors(expression, axis)
            value = (left+right)/2
        elif is_diff(expression.op):
            normal = self.block.axes[axis]
            letters = expression.op[1:]
            if normal in letters:
                remaining = letters.replace(normal, '', 1)
                differentiated = make('D'+remaining, expression.args[0]) if remaining else expression.args[0]
                left, right = self._expression_neighbors(differentiated, axis)
                value = (right - left) / self._spacing(axis)
            else:
                left, right = self._expression_neighbors(expression, axis)
                value = (left+right)/2
        else:
            args = [self.face(child, axis) for child in expression.args]
            if expression.op == 'Add': value = args[0] + args[1]
            elif expression.op == 'Mul': value = args[0] * args[1]
            elif expression.op == 'Div': value = args[0] / args[1]
            else: value = {'Sin': torch.sin, 'Cos': torch.cos, 'Tanh': torch.tanh}[expression.op](args[0])
        self.face_cache[key] = value
        self.face_nodes_computed += 1
        return value

    def face_batch(self, expressions, axis):
        return torch.cat([self.face(expression, axis) for expression in expressions], dim=-1)

    def _face_observation_mask(self, axis):
        """Return zero/one support at the same faces as ``face``.

        The mask is derived only from recorded observations.  Zero ghost
        support prevents an unobserved exterior face from entering a masked
        weak projection.
        """
        block = self.block
        if block.observation_mask is None or block.full_observation:
            return None
        mask = block.observation_mask.reshape(*block.shape)
        if self._periodic(axis):
            left, right = self._neighbors(mask, axis)
        else:
            edge = torch.zeros_like(mask.narrow(axis, 0, 1))
            padded = torch.cat((edge, mask, edge), dim=axis)
            first = [slice(None)] * mask.ndim
            second = first.copy()
            first[axis], second[axis] = slice(None, -1), slice(1, None)
            left, right = padded[tuple(first)], padded[tuple(second)]
        return ((left + right) / 2).unsqueeze(-1)

    def project_face_flux(self, flux, axis):
        """Integrate a discrete face divergence with its exact adjoint stencil."""
        face_mask = self._face_observation_mask(axis)
        value = flux if face_mask is None else flux * face_mask
        for current, weights in enumerate(self.block.weights):
            matrix = weights[0]
            if current == axis:
                if not self._periodic(axis) and self.boundary_extensions is not None:
                    matrix = torch.nn.functional.pad(matrix, (1, 1))
                difference = (torch.roll(matrix, -1, dims=1) - matrix if self._periodic(axis)
                              else matrix[:, 1:] - matrix[:, :-1])
                matrix = -difference / self._spacing(axis)
            value = torch.tensordot(matrix, value, dims=([1], [current])).movedim(0, current)
        value = value.reshape(self.block.full_rows, -1)
        if self.block.observation_mask is not None and not self.block.full_observation:
            # Match Block.project: normalize by observed quadrature coverage
            # and retain only windows that passed the support check.
            value = value / self.block.observed_coverage.clamp_min(1e-30)
            if self.block.selected_indices is not None:
                value = value[self.block.selected_indices]
        elif self.block.row_indices is not None:
            value = value[self.block.row_indices]
        return value

    def _weak_one(self, expression):
        if expression in self.conservative_weak_cache:
            return self.conservative_weak_cache[expression].to(self.block.device)
        if expression.op == 'Add':
            value = self._weak_one(expression.args[0]) + self._weak_one(expression.args[1])
        elif is_diff(expression.op):
            letters = expression.op[1:]
            axis = self.block.axes.index(letters[0])
            child = expression.args[0]
            if len(letters) > 1:
                child = make('D' + letters[1:], child)
            value = self.project_face_flux(self.face(child, axis), axis)
        else:
            value = self.block.project(self.field(expression))
        self.conservative_weak_cache[expression] = value
        self.projected += 1
        return value

    def weak(self, expressions):
        return torch.cat([self._weak_one(expression) for expression in expressions], dim=1)

    def weak_direct(self, expressions):
        return self.weak(expressions)

    def weak_stream(self, expressions, compiled_plan=None):
        # Preserve face-centred summation by parts while bounding temporary
        # cell/face values. Only prefixes present on entry survive a batch.
        names = ('cache', 'weak_cache', 'face_cache', 'conservative_weak_cache')
        permanent = {name: getattr(self, name) for name in names}
        outputs = []
        try:
            # Keep face projections large enough to amortize GPU launch cost.
            for start in range(0, len(expressions), 256):
                for name in names:
                    setattr(self, name, permanent[name].copy())
                outputs.append(self.weak(expressions[start:start + 256]))
            result = torch.cat(outputs, dim=1)
            if result.shape[1] != len(expressions):
                raise RuntimeError('Conservative streaming changed candidate column count')
            return result
        finally:
            for name in names:
                setattr(self, name, permanent[name])

    def retain(self, expressions, compact=True):
        super().retain(expressions, compact=compact)
        wanted = set()
        def visit(expression):
            if expression in wanted: return
            wanted.add(expression)
            for child in expression.args: visit(child)
        for expression in expressions: visit(expression)
        transform=self._compact if compact else (lambda value:value)
        self.face_cache = {key: transform(value) for key, value in self.face_cache.items() if key[0] in wanted}
        self.conservative_weak_cache = {key: transform(value) for key, value in self.conservative_weak_cache.items() if key in wanted}

    def offload(self):
        super().offload()
        self.face_cache = {key: value.cpu() for key, value in self.face_cache.items()}
        self.conservative_weak_cache = {key: value.cpu() for key, value in self.conservative_weak_cache.items()}

    def clear(self):
        self.cache.clear(); self.weak_cache.clear()
        self.face_cache.clear(); self.conservative_weak_cache.clear()

    def prune_to(self, permanent_nodes):
        """Release candidate batch values while retaining shared prefixes."""
        self.retain(permanent_nodes, compact=False)

    def release_values(self):
        """Release every value cache before model selection or block disposal."""
        self.clear()
