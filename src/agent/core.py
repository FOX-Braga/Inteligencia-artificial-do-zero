from typing import List, Dict, Any
from src.llm.generator import LocalGenerator
from src.agent.state import AgentState
from src.tools.base import BaseTool

class LocalAgent:
    """
    Agente que usa o nosso modelo LLM treinado do zero localmente.
    """
    def __init__(self, generator: LocalGenerator, tools: List[BaseTool] = None):
        self.generator = generator
        self.state = AgentState()
        self.tools = {tool.name: tool for tool in (tools or [])}
        
    def add_tool(self, tool: BaseTool):
        self.tools[tool.name] = tool
        
    def run(self, user_input: str) -> str:
        """
        Executa um ciclo simples de inferência.
        Num modelo criado do zero muito pequeno, o Agent não será inteligente o 
        suficiente para usar ferramentas (Function Calling / ReAct) sem muito treinamento
        e fine-tuning especializado. 
        Então, vamos focar apenas em responder o prompt diretamente.
        """
        self.state.add_message("user", user_input)
        
        # Constrói o prompt para o modelo ler as mensagens anteriores
        prompt = ""
        for msg in self.state.get_messages():
            prompt += f"\n{msg['role']}: {msg['content']}"
            
        prompt += "\nassistant:"
        
        # Gera a resposta do modelo
        response_text = self.generator.generate_text(prompt, max_new_tokens=200)
        
        self.state.add_message("assistant", response_text.strip())
        return response_text.strip()
