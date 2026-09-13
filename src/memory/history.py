import json
import os
from typing import List, Dict, Any

class ConversationHistory:
    def __init__(self, file_path: str = "history.json"):
        self.file_path = file_path
        
    def save(self, messages: List[Dict[str, Any]]):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
            
    def load(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.file_path):
            return []
            
        with open(self.file_path, "r", encoding="utf-8") as f:
            return json.load(f)
