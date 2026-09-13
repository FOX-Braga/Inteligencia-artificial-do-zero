import pygame
import random
import math
import sys

# Cores no estilo da imagem
BACKGROUND = (10, 10, 20)
CYAN = (0, 200, 255)
CYAN_DIM = (0, 50, 100)
ORANGE = (255, 100, 0)
ORANGE_DIM = (100, 40, 0)
WHITE = (255, 255, 255)
LINE_COLOR = (0, 100, 150)

class NeuralNetVisualizer:
    def __init__(self, width=1024, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Treinamento da Rede Neural - Visualizador")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Definir uma arquitetura visual genérica (apenas representativa)
        # Camadas: 1 Entrada, 3 Ocultas, 1 Saída
        self.layers = [15, 25, 30, 25, 15] 
        self.nodes = [] # Lista de posições de cada nó
        
        self._calculate_node_positions()
        
    def _calculate_node_positions(self):
        """ Calcula as posições X, Y de cada neurônio na tela """
        layer_spacing_x = self.width / (len(self.layers) + 1)
        
        for layer_idx, num_nodes in enumerate(self.layers):
            layer_nodes = []
            x = layer_spacing_x * (layer_idx + 1)
            
            node_spacing_y = self.height / (num_nodes + 1)
            for node_idx in range(num_nodes):
                y = node_spacing_y * (node_idx + 1)
                
                # Deslocamento sutil em X para o efeito cônico da imagem
                if layer_idx == 0:
                    x_offset = -30
                elif layer_idx == len(self.layers) - 1:
                    x_offset = 30
                else:
                    x_offset = 0
                    
                layer_nodes.append((x + x_offset, y))
            self.nodes.append(layer_nodes)
            
        self._pre_render_background()
        
    def _pre_render_background(self):
        """ OTIMIZAÇÃO: Desenha as linhas estáticas de fundo de uma vez só numa Surface """
        self.bg_surface = pygame.Surface((self.width, self.height))
        self.bg_surface.fill(BACKGROUND)
        
        for i in range(len(self.layers) - 1):
            current_layer = self.nodes[i]
            next_layer = self.nodes[i+1]
            base_color = (80, 25, 0) if i == 0 else (0, 40, 60)
            
            for node_a in current_layer:
                for node_b in next_layer:
                    pygame.draw.line(self.bg_surface, base_color, node_a, node_b, 1)

    def draw_connections(self, active_fraction=0.1):
        """ OTIMIZAÇÃO: Desenha apenas as sinapses brilhantes selecionando uma amostra """
        for i in range(len(self.layers) - 1):
            current_layer = self.nodes[i]
            next_layer = self.nodes[i+1]
            active_color = ORANGE if i == 0 else CYAN
            
            # Sorteia apenas uma fração dos nós para piscar
            num_active = int(len(current_layer) * active_fraction)
            if num_active == 0: continue
            
            active_nodes_a = random.sample(current_layer, num_active)
            for node_a in active_nodes_a:
                # Cada nó ativo se conecta a no máximo 2 nós da próxima camada para poupar processamento
                targets = random.sample(next_layer, min(2, len(next_layer)))
                for node_b in targets:
                    pygame.draw.line(self.screen, active_color, node_a, node_b, 2)

    def draw_nodes(self, active_fraction=0.3):
        """ Desenha os neurônios (círculos) """
        for i, layer in enumerate(self.nodes):
            is_input = (i == 0)
            
            for node in layer:
                # Efeito de "ativação" (piscar)
                is_active = random.random() < active_fraction
                
                if is_input:
                    color = ORANGE if is_active else ORANGE_DIM
                else:
                    color = CYAN if is_active else CYAN_DIM
                    
                # Desenhar brilho (halo)
                if is_active:
                    pygame.draw.circle(self.screen, color, node, 6)
                
                # Desenhar centro (branco/brilhante)
                center_color = WHITE if is_active else color
                pygame.draw.circle(self.screen, center_color, node, 3)

    def update(self, loss: float, step: int, input_text: str = "", output_text: str = ""):
        """ Método chamado a cada passo do treinamento para atualizar a tela """
        
        # Processar eventos para a janela não travar (ex: fechar janela)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # OTIMIZAÇÃO: Usa o background pre-renderizado (copia muito mais rápido do que redesenhar 2250 linhas)
        self.screen.blit(self.bg_surface, (0, 0))
        
        # Efeito visual de que a rede está processando
        # Quanto menor a loss, mais estável/focado fica o brilho (active_fraction)
        activity = max(0.05, min(0.4, loss / 10.0))
        
        self.draw_connections(active_fraction=activity)
        self.draw_nodes(active_fraction=activity + 0.1)
        
        # Renderizar Textos HUD
        loss_text = self.font.render(f"Loss: {loss:.4f} | Passo: {step}", True, WHITE)
        self.screen.blit(loss_text, (20, 20))
        
        if input_text and output_text:
            in_text = self.font.render(f"Entrada: {input_text[:40]}...", True, ORANGE)
            out_text = self.font.render(f"Saída:   {output_text[:40]}...", True, CYAN)
            self.screen.blit(in_text, (20, self.height - 80))
            self.screen.blit(out_text, (20, self.height - 50))
            
        pygame.display.flip()
        
        # OTIMIZAÇÃO: Limitar a 15 FPS para deixar a CPU livre pro PyTorch treinar
        self.clock.tick(15)
