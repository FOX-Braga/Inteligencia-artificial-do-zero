from typing import List, Dict, Any

class AgentState:
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        
    def add_tool_result(self, tool_use_id: str, content: str):
        self.messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": content
                }
            ]
        })
        
    def get_messages(self) -> List[Dict[str, Any]]:
        return self.messages
        
    def clear(self):
        self.messages = []
