import pygame
import random
import sys

# Cores baseadas em verde neon
BACKGROUND = (5, 5, 5)
GREEN_BRIGHT = (0, 255, 100)
GREEN_DIM = (0, 80, 30)
WHITE = (220, 255, 220)
DIM_GRAY = (20, 50, 20)

class NeuralNetVisualizer:
    def __init__(self, width=1024, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Treinamento da Rede Neural - Visualizador")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 20)
        self.large_font = pygame.font.SysFont("Consolas", 28)
        
        # Arquitetura exata da imagem: Entrada (gap no meio), 2 Ocultas Densas, 1 Saída (10 nós)
        self.layers = [15, 14, 14, 10] 
        self.nodes = [] 
        
        self._calculate_node_positions()
        
    def _calculate_node_positions(self):
        """ Calcula as posições X, Y de cada neurônio na tela """
        layer_spacing_x = self.width / (len(self.layers) + 1)
        
        for layer_idx, num_nodes in enumerate(self.layers):
            layer_nodes = []
            x = layer_spacing_x * (layer_idx + 1)
            
            # Margens para centralizar verticalmente
            margin_y = 100 if layer_idx < 3 else 150
            usable_height = self.height - 2 * margin_y
            node_spacing_y = usable_height / max(1, num_nodes - 1)
            
            for node_idx in range(num_nodes):
                # Para a camada de entrada, pulamos o nó central para desenhar os "..."
                if layer_idx == 0 and node_idx == num_nodes // 2:
                    continue
                    
                y = margin_y + node_idx * node_spacing_y
                layer_nodes.append((x, y))
                
            self.nodes.append(layer_nodes)
            
        self._pre_render_background()
        
    def _pre_render_background(self):
        """ Desenha as linhas estáticas de fundo (todas as conexões em cinza escuro) """
        self.bg_surface = pygame.Surface((self.width, self.height))
        self.bg_surface.fill(BACKGROUND)
        
        for i in range(len(self.layers) - 1):
            current_layer = self.nodes[i]
            next_layer = self.nodes[i+1]
            
            for node_a in current_layer:
                for node_b in next_layer:
                    pygame.draw.line(self.bg_surface, DIM_GRAY, node_a, node_b, 1)

    def draw_connections(self, active_fraction=0.1):
        """ Desenha apenas as sinapses coloridas (Laranja/Ciano) que estão ativas """
        for i in range(len(self.layers) - 1):
            current_layer = self.nodes[i]
            next_layer = self.nodes[i+1]
            
            # Na imagem, a primeira camada de pesos tem muitas linhas laranjas, as outras ciano/brancas
            active_color = GREEN_BRIGHT
            
            # Sorteia uma fração dos nós
            num_active = int(len(current_layer) * active_fraction)
            if num_active == 0: continue
            
            active_nodes_a = random.sample(current_layer, num_active)
            for node_a in active_nodes_a:
                targets = random.sample(next_layer, min(3, len(next_layer)))
                for node_b in targets:
                    # Linhas ativas um pouco mais finas, mas brilhantes
                    pygame.draw.line(self.screen, active_color, node_a, node_b, 1)

    def draw_nodes(self, active_fraction=0.3):
        """ Desenha os neurônios (círculos ocos) idênticos à imagem """
        for i, layer in enumerate(self.nodes):
            for node_idx, node in enumerate(layer):
                # Fundo preto para encobrir as linhas que passam por trás
                pygame.draw.circle(self.screen, BACKGROUND, node, 9)
                # Borda branca/cinza claro
                pygame.draw.circle(self.screen, WHITE, node, 9, 1)
                
                # Efeito de "ativação" (preenchimento sutil)
                if random.random() < active_fraction:
                    pygame.draw.circle(self.screen, GREEN_BRIGHT, node, 5)
                    
            # --- Elementos Visuais Específicos da Imagem ---
            if i == 0:
                # 1. Os 3 pontos (...) no meio da camada de entrada
                x = layer[0][0]
                mid_y = self.height / 2
                for dy in [-12, 0, 12]:
                    pygame.draw.circle(self.screen, WHITE, (x, mid_y + dy), 2)
                    
                # 2. A chave "{" gigante
                x_brace = x - 25
                y_top = layer[0][1]
                y_bottom = layer[-1][1]
                pygame.draw.line(self.screen, WHITE, (x_brace, y_top), (x_brace - 5, y_top), 2)
                pygame.draw.line(self.screen, WHITE, (x_brace - 5, y_top), (x_brace - 5, mid_y - 5), 2)
                pygame.draw.line(self.screen, WHITE, (x_brace - 5, mid_y - 5), (x_brace - 15, mid_y), 2)
                pygame.draw.line(self.screen, WHITE, (x_brace - 15, mid_y), (x_brace - 5, mid_y + 5), 2)
                pygame.draw.line(self.screen, WHITE, (x_brace - 5, mid_y + 5), (x_brace - 5, y_bottom), 2)
                pygame.draw.line(self.screen, WHITE, (x_brace - 5, y_bottom), (x_brace, y_bottom), 2)
                
                # 3. O texto "784"
                label_text = self.large_font.render("784", True, WHITE)
                self.screen.blit(label_text, (x_brace - 75, mid_y - 15))
                
            elif i == len(self.nodes) - 1:
                # 4. Labels 0 a 9 na camada de saída
                for node_idx, node in enumerate(layer):
                    label = self.font.render(str(node_idx), True, WHITE)
                    self.screen.blit(label, (node[0] + 20, node[1] - 10))

    def run(self, shared_state: dict):
        """
        Loop principal da GUI — roda na thread principal.
        Lê os dados de 'shared_state' que a thread de treinamento atualiza.
        """
        while True:
            # ── Processar eventos do sistema operacional ─────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # ── Ler dados do treinamento (thread segura pois é leitura simples) ─
            loss        = shared_state.get("loss", 0.0)
            step        = shared_state.get("step", 0)
            input_text  = shared_state.get("input_text", "")
            output_text = shared_state.get("output_text", "")
            running     = shared_state.get("running", True)

            # ── Desenhar ─────────────────────────────────────────────────────
            self.screen.blit(self.bg_surface, (0, 0))

            activity = max(0.05, min(0.4, loss / 10.0)) if loss > 0 else 0.1
            self.draw_connections(active_fraction=activity)
            self.draw_nodes(active_fraction=activity + 0.1)

            # HUD — Cabeçalho
            status = "Treinando..." if running else "✔ Concluído!"
            loss_text = self.font.render(f"Loss: {loss:.4f} | Passo: {step} | {status}", True, WHITE)
            self.screen.blit(loss_text, (20, 20))

            # HUD — Rodapé
            if input_text and output_text:
                in_text  = self.font.render(f"Entrada: {input_text[:50]}...",  True, GREEN_BRIGHT)
                out_text = self.font.render(f"Saída:   {output_text[:50]}...", True, GREEN_DIM)
                self.screen.blit(in_text,  (20, self.height - 80))
                self.screen.blit(out_text, (20, self.height - 50))

            pygame.display.flip()
            self.clock.tick(60)   # 60 FPS contínuo sem travar

