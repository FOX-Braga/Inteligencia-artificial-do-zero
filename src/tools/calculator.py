from src.tools.base import BaseTool
from typing import Any, Dict
import ast
import operator

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Executa expressões matemáticas simples."
    
    def __init__(self):
        # Operadores permitidos
        self.operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg
        }
        
    def _eval(self, node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return self.operators[type(node.op)](self._eval(node.left), self._eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            return self.operators[type(node.op)](self._eval(node.operand))
        else:
            raise TypeError(node)

    def execute(self, expression: str) -> str:
        try:
            tree = ast.parse(expression, mode='eval')
            result = self._eval(tree.body)
            return str(result)
        except Exception as e:
            return f"Erro ao calcular: {e}"
            
    def get_tool_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A expressão matemática a ser avaliada (ex: '2 + 2')"
                    }
                },
                "required": ["expression"]
            }
        }
