import math
import random
import sys
from collections import deque

import pygame

# ── Paleta (verde neon, estilo original) ─────────────────────────────────────
BACKGROUND   = (5, 5, 5)
GREEN_BRIGHT = (0, 255, 100)
GREEN_MID    = (0, 170, 70)
GREEN_DIM    = (0, 80, 30)
WHITE        = (220, 255, 220)
DIM_GRAY     = (24, 48, 24)
PANEL_BG     = (10, 16, 10)
BORDER       = (40, 70, 40)
YELLOW       = (255, 220, 90)
RED          = (255, 90, 90)
CYAN         = (90, 220, 255)


def _lerp_color(a, b, t):
    """Interpola cor a -> b com t em [0, 1]."""
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


class NeuralNetVisualizer:
    def __init__(self, width=1280, height=760):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("IA do Zero - Rede Neural em Treinamento")
        pygame.display.set_icon(self.screen)

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 18)
        self.small_font = pygame.font.SysFont("Consolas", 14)
        self.big_font = pygame.font.SysFont("Consolas", 30)

        # Arquitetura visual da rede: Entrada, 2 Ocultas, Saída
        self.layer_names = ["ENTRADA", "OCULTA 1", "OCULTA 2", "SAIDA"]
        self.layers = [15, 14, 14, 10]
        self.nodes = []
        self._area = None          # retângulo (x,y,w,h) da rede
        self._calculate_node_positions()
        self._pre_render_background()

        # Camada da rede renderizada OFFscreen: só é redesenhada quando as
        # ativações mudam (a cada passo do treino, ~0,8s). Entre passos são
        # apenas *blits* leves, o que mantém 60 FPS mesmo com a GPU ocupada.
        self._net_layer = pygame.Surface((self.width, self.height))
        self._net_layer.blit(self.bg_surface, (0, 0))
        self._net_sig = None

        # Histórico do loss (sparkline) — mantido aqui (não cruza threads)
        self.loss_history = deque(maxlen=200)
        self._last_step = -1
        self._spark = None

        # Caches antigarantia: renderizar fonte é o maior custo da GUI;
        # guarda as superfícies de texto e as quebras de linha prontas.
        self._text_cache = {}
        self._wrap_cache = {}
        self._MAX_TEXT_CACHE = 4096

    # ── Geometria ────────────────────────────────────────────────────────────
    def _calculate_node_positions(self):
        area_x = 90
        area_y = 190
        area_w = self.width - area_x - 60
        area_h = self.height - area_y - 200

        layer_spacing = area_w / (len(self.layers) + 1)
        layer_nodes = []
        for layer_idx, num in enumerate(self.layers):
            nodes_here = []
            x = area_x + layer_spacing * (layer_idx + 1)
            usable = area_h - 30
            spacing_y = usable / max(1, num - 1)
            for n in range(num):
                y = area_y + 15 + n * spacing_y
                nodes_here.append((x, y))
            layer_nodes.append(nodes_here)
        self.nodes = layer_nodes
        self._area = (area_x, area_y, area_w, area_h)

    def _pre_render_background(self):
        """Desenha a malha cinza estática + painéis + títulos num surface fixo."""
        surf = pygame.Surface((self.width, self.height))
        surf.fill(BACKGROUND)

        # Malha de conexões (todas em cinza escuro)
        for i in range(len(self.layers) - 1):
            for a in self.nodes[i]:
                for b in self.nodes[i + 1]:
                    pygame.draw.line(surf, DIM_GRAY, a, b, 1)

        # Nós "esqueleto"
        for layer in self.nodes:
            for (x, y) in layer:
                pygame.draw.circle(surf, BACKGROUND, (x, y), 15)
                pygame.draw.circle(surf, BORDER, (x, y), 15, 1)

        # Títulos das camadas
        for i, name in enumerate(self.layer_names):
            x = self.nodes[i][0][0]
            text = self.small_font.render(name, True, GREEN_DIM)
            surf.blit(text, (x - text.get_width() // 2,
                             self.nodes[i][0][1] - 40))

        # Painel de entrada/saída
        pygame.draw.rect(surf, PANEL_BG, (20, self.height - 165, self.width - 40, 145))
        pygame.draw.rect(surf, BORDER, (20, self.height - 165, self.width - 40, 145), 1)

        # Painel de métricas (topo)
        pygame.draw.rect(surf, PANEL_BG, (20, 20, self.width - 40, 110))
        pygame.draw.rect(surf, BORDER, (20, 20, self.width - 40, 110), 1)

        self.bg_surface = surf

    # ── Caches de texto (60 FPS sem re-render de fonte) ──────────────────────
    def _render_text(self, text, font, color):
        """Retorna uma superfície de texto cacheada por (fonte, texto, cor)."""
        key = (id(font), text, color)
        img = self._text_cache.get(key)
        if img is None:
            if len(self._text_cache) >= self._MAX_TEXT_CACHE:
                self._text_cache.clear()
            img = font.render(text, True, color)
            self._text_cache[key] = img
        return img

    def _blit(self, text, x, y, color=WHITE, font=None, aa=True):
        font = font or self.font
        img = self._render_text(text, font, color)
        self.screen.blit(img, (x, y))
        return img

    def _blit_wrap(self, text, x, y, max_w, max_h, color=WHITE, font=None):
        """Desenha texto com quebra de linha, retornando a altura final usada."""
        font = font or self.font
        key = (id(font), text, max_w)
        lines = self._wrap_cache.get(key)
        if lines is None:
            words = text.split(" ")
            lines, line = [], ""
            for w in words:
                trial = (line + " " + w).strip()
                if font.size(trial)[0] <= max_w:
                    line = trial
                else:
                    if line:
                        lines.append(line)
                    line = w
            if line:
                lines.append(line)
            if len(self._wrap_cache) > 512:
                self._wrap_cache.clear()
            self._wrap_cache[key] = lines
        yy = y
        for ln in lines[: max(1, int(max_h / (font.get_height() + 2)))]:
            self._blit(ln, x, yy, color, font)
            yy += font.get_height() + 2
        return yy - y

    # ── Desenho da rede com ativações REAIS ──────────────────────────────────
    def _node_color(self, act):
        """Cor do nó conforme ativação (0→verde escuro, 1→verde neon)."""
        return _lerp_color((8, 25, 10), GREEN_BRIGHT, act)

    def _draw_active_network(self, values):
        """
        Pinta os neurônios e conexões de acordo com as ativações reais.
        `values` é uma lista de 4 listas: [inputs, hidden1, hidden2, outputs].
        Desenha na camada offscreen (`_net_layer`), que o loop só recria quando
        os valores mudam.
        """
        target = self._net_layer
        activations = []
        for layer_vals in values:
            if layer_vals:
                activations.append(layer_vals)
            else:
                activations.append([0.0] * self.layers[len(activations)])

        # Conexões ativas: de cada nó com ativação alta para os nós fortes da
        # próxima camada.
        for i in range(len(self.layers) - 1):
            src = self.nodes[i]
            dst = self.nodes[i + 1]
            a_src = activations[i]
            a_dst = activations[i + 1]
            strong_targets = sorted(range(len(dst)), key=lambda j: a_dst[j],
                                    reverse=True)[:3]
            for s in range(len(src)):
                if a_src[s] >= 0.35:
                    for t in strong_targets:
                        strength = (a_src[s] + a_dst[t]) / 2
                        color = _lerp_color(DIM_GRAY, GREEN_BRIGHT, strength)
                        pygame.draw.line(target, color, src[s], dst[t], 1)

        # Nós coloridos (preenchimento proporcional à ativação real)
        for li, layer in enumerate(self.nodes):
            for (x, y), act in zip(layer, activations[li]):
                pygame.draw.circle(target, BACKGROUND, (x, y), 14)
                color = self._node_color(act)
                pygame.draw.circle(target, color, (x, y), 14, 0)
                pygame.draw.circle(target, WHITE, (x, y), 14, 1)
                # Valor numérico dentro/abaixo do nó
                label = self._render_text(f"{act:.1f}", self.small_font, (0, 12, 5))
                target.blit(label, (x - label.get_width() // 2,
                                    y - label.get_height() // 2))

        # Rótulos dos tokens na camada de SAÍDA (o que o modelo prevê)
        outputs = self._last_outputs or []
        for (x, y), (tok, prob) in zip(self.nodes[-1], outputs):
            txt = self._clean(tok)

            def _ts(s, f=None):
                f = f or self.small_font
                return self._render_text(s, f, CYAN)

            t_img = _ts(txt)
            p_img = _ts(f"{prob:.2%}", self.small_font)
            target.blit(t_img, (x + 18, y - 11))
            target.blit(p_img, (x + 18, y + 2))

    def _clean(self, s):
        """Sanitiza um token para exibição."""
        s = s.replace("\n", "\\n").replace(" ", "_")
        return s[:10] if s else "_"

    def _sig(self, state):
        """Assinatura leve (hash barato) do que a rede exibe num dado momento.
        Se não mudou entre frames, o desenho da rede é reaproveitado."""
        def snap(vals, scale=100):
            return tuple(int(round(v * scale)) for v in vals)
        activ = (snap(state.get("inputs", [])),
                 snap(state.get("hidden1", [])),
                 snap(state.get("hidden2", [])))
        outputs = tuple((str(tok), int(round(float(prob) * 10000)))
                        for tok, prob in (self._last_outputs or []))
        return activ, outputs

    # ── Métricas / cabeçalho ─────────────────────────────────────────────────
    def _draw_header(self, state):
        running = state.get("running", True)
        loss = state.get("loss", 0.0)
        err = state.get("error_rate", 1.0)
        acc = state.get("acc", 0.0)
        ppl = state.get("perplexity", 0.0)
        step = state.get("step", 0)
        total = state.get("total_steps", 1) or 1

        title = self._render_text("IA do Zero - Treinamento", self.big_font, GREEN_BRIGHT)
        self.screen.blit(title, (40, 30))

        status = state.get("status") or ("TREINANDO..." if running else "CONCLUIDO")
        if status.startswith("ERRO"):
            s_color = RED
        elif status.startswith(("TREIN", "CONCL")):
            s_color = GREEN_BRIGHT
        else:
            s_color = YELLOW
        s_font = self.big_font if status.startswith(("TREIN", "CONCL", "ERRO")) else self.small_font
        self._blit(status, 40, 68, s_color, s_font)

        # Cartões de métrica
        cards = [
            ("LOSS", f"{loss:.4f}", WHITE),
            ("ERRO", f"{err:.1%}", RED if err > 0.5 else YELLOW),
            ("ACERTO", f"{acc:.1%}", GREEN_BRIGHT),
            ("PERPLEX", f"{ppl:.1f}" if ppl else "-", CYAN),
            ("TENTATIVA", f"{step} / {total}", YELLOW),
        ]
        x = 330
        card_w = 165
        for name, value, color in cards:
            pygame.draw.rect(self.screen, (12, 20, 12), (x, 30, card_w, 72))
            pygame.draw.rect(self.screen, BORDER, (x, 30, card_w, 72), 1)
            self._blit(name, x + 10, 36, GREEN_DIM, self.small_font)
            val = self._render_text(value, self.font, color)
            self.screen.blit(val, (x + 10, 58))
            x += card_w + 12

        # Barra de progresso
        prog = step / total
        bx = 40
        by = 140
        bw = self.width - 80
        pygame.draw.rect(self.screen, (12, 18, 12), (bx, by, bw, 14))
        fill_w = int(bw * prog)
        if fill_w > 0:
            pygame.draw.rect(self.screen, GREEN_MID, (bx, by, fill_w, 14))
        pygame.draw.rect(self.screen, BORDER, (bx, by, bw, 14), 1)
        self._blit(f"{prog:.1%} concluido", bx, by + 18, GREEN_DIM, self.small_font)

    # ── Rodapé: entrada/saída + gráfico ──────────────────────────────────────
    def _draw_footer(self, state):
        in_text = state.get("input_text", "")
        out_text = state.get("output_text", "")
        y0 = self.height - 155
        max_w = self.width - 460

        self._blit("ENTRADA (contexto real)", 35, y0, GREEN_MID, self.small_font)
        if in_text:
            self._blit_wrap(">" + in_text[:400], 35, y0 + 18, max_w, 50, WHITE, self.small_font)

        self._blit("SAIDA (o que o modelo previu)", 35, y0 + 90, GREEN_BRIGHT, self.small_font)
        if out_text:
            self._blit_wrap("> " + out_text[:400], 35, y0 + 108, max_w, 50, GREEN_BRIGHT, self.small_font)

        # Sparkline do loss (pré-renderizado; só reconstrói ao mudar o passo)
        gx = self.width - 280
        gy = y0 - 10
        gw = 240
        gh = 60
        pygame.draw.rect(self.screen, (10, 14, 10), (gx, gy, gw, gh))
        pygame.draw.rect(self.screen, BORDER, (gx, gy, gw, gh), 1)
        if self._spark is not None:
            self.screen.blit(self._spark, (gx, gy))
        self._blit("LOSS", gx + 6, gy - 2, GREEN_DIM, self.small_font)

    def _render_spark(self):
        """Reconstrói a superfície do sparkline (chamada só quando o loss muda)."""
        gw, gh = 240, 60
        surf = pygame.Surface((gw, gh), pygame.SRCALPHA)
        pts = list(self.loss_history)
        if len(pts) > 1:
            lo, hi = min(pts), max(pts)
            rng = (hi - lo) or 1.0
            n = len(pts)
            for i in range(1, n):
                x1 = 6 + (i - 1) * (gw - 12) / (n - 1)
                y1 = gh - 8 - (pts[i - 1] - lo) / rng * (gh - 18)
                x2 = 6 + i * (gw - 12) / (n - 1)
                y2 = gh - 8 - (pts[i] - lo) / rng * (gh - 18)
                pygame.draw.line(surf, GREEN_BRIGHT, (x1, y1), (x2, y2), 2)
        self._spark = surf

    # ── Loop principal ───────────────────────────────────────────────────────
    def run(self, shared_state: dict):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Atualiza histórico do loss quando muda de passo
            step = shared_state.get("step", 0)
            if step != self._last_step:
                self._last_step = step
                self.loss_history.append(shared_state.get("loss", 0.0))
                self._render_spark()

            self._last_outputs = shared_state.get("outputs", [])

            # Só redesenha a rede quando as ativações/tokens mudam (a cada
            # passo de treino); nos frames intermediários reusa a camada.
            sig = self._sig(shared_state)
            if sig != self._net_sig:
                self._net_sig = sig
                self._net_layer.blit(self.bg_surface, (0, 0))
                self._draw_active_network([
                    shared_state.get("inputs", []),
                    shared_state.get("hidden1", []),
                    shared_state.get("hidden2", []),
                    [o[1] if isinstance(o, (tuple, list)) else o
                     for o in self._last_outputs],
                ])

            self.screen.blit(self.bg_surface, (0, 0))
            self.screen.blit(self._net_layer, (0, 0))
            self._draw_header(shared_state)
            self._draw_footer(shared_state)

            pygame.display.flip()
            self.clock.tick(60)   # 60 FPS (texto cacheado mantém o render leve)