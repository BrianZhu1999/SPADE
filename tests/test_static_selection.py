"""Exercise final static selection without running candidate enumeration."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class StaticSelectionTest(unittest.TestCase):
    def test_empty_candidates_report_selection_failure(self):
        source = Path(__file__).resolve().parents[1] / 'run_search.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        main = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'main')
        # Execute the production selection loop after the expensive search stage.
        loop = next(node for node in main.body
                    if isinstance(node, ast.For)
                    and isinstance(node.iter, ast.Call)
                    and isinstance(node.iter.func, ast.Name)
                    and node.iter.func.id == 'enumerate'
                    and len(node.iter.args) == 1
                    and isinstance(node.iter.args[0], ast.Name)
                    and node.iter.args[0].id == 'paths')
        function = ast.parse('def check_selection():\n    pass\n').body[0]
        function.body = [loop]
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        code = compile(module, str(source), 'exec')
        for fast in (False, True):
            with self.subTest(fast_final_selection=fast):
                namespace = {
                    'a': SimpleNamespace(require_spatial_term=False,
                                         fast_final_selection=fast),
                    'static': True,
                    'paths': [[]],
                    'exprs': [],
                    'SelectionAlgebra': lambda expressions: None,
                }
                exec(code, namespace)
                with self.assertRaisesRegex(
                        RuntimeError,
                        '^No spatial differential equation retained for steady observations$'):
                    namespace['check_selection']()


if __name__ == '__main__':
    unittest.main()
