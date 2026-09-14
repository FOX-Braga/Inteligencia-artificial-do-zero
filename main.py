from src.llm.generator import LocalGenerator
from src.agent.core import LocalAgent
from src.tools.calculator import CalculatorTool
from src.tools.search import SearchTool
from config import Config

def main():
    print("Inicializando o Agente de IA (Modelo Local criado do Zero)...")
    
    # 1. Inicializar o gerador (carrega os pesos e o vocabulário localmente)
    # Se o modelo não foi treinado, ele vai gerar um aviso e usar pesos aleatórios
    generator = LocalGenerator(model_path=Config.MODEL_PATH, vocab_path=Config.VOCAB_PATH)
        
    # 2. Inicializar Tools (As ferramentas estão aqui, mas nosso modelo pequeno
    # recém-criado precisaria de muito fine-tuning pra aprender a usá-las)
    tools = [
        CalculatorTool(),
        SearchTool()
    ]
    
    # 3. Criar Agente
    agent = LocalAgent(generator=generator, tools=tools)
    
    print("\nAgente pronto! Digite 'sair' para encerrar.")
    print("-" * 50)
    
    # 4. Loop de conversação
    while True:
        try:
            user_input = input("\nVocê: ")
            if user_input.lower() in ['sair', 'exit', 'quit']:
                break
                
            if not user_input.strip():
                continue
                
            print("Gerando...")
            response = agent.run(user_input)
            
            print(f"\nIA Local: {response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nErro de execução: {e}")

if __name__ == "__main__":
    main()
