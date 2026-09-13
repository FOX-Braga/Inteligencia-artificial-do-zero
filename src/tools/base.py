from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseTool(ABC):
    name: str
    description: str
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """
        Executa a ferramenta com os argumentos passados pelo Claude.
        """
        pass
        
    def get_tool_schema(self) -> Dict[str, Any]:
        """
        Retorna o schema no formato esperado pela API do Claude para tool_use.
        Deve ser implementado nas subclasses.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
