import torch


class AdamW_DML(torch.optim.Optimizer):
    """
    AdamW com weight decay desacoplado, implementado apenas com operações
    elementares que o backend DirectML (AMD no Windows) suporta nativamente.

    O AdamW original do PyTorch usa operações `_foreach_lerp_` que NÃO são
    suportadas no DirectML e caem para a CPU a cada passo (muito mais lento).
    Esta implementação evita esse fallback.

    Fórmula (AdamW — Loshchilov & Hutter 2019):
        t        = t + 1
        m        = beta1 * m + (1 - beta1) * g
        v        = beta2 * v + (1 - beta2) * g^2
        m_hat    = m / (1 - beta1^t)
        v_hat    = v / (1 - beta2^t)
        theta   -= lr * (theta * wd + m_hat / (sqrt(v_hat) + eps))
    """
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0, dtype=torch.long, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Correção de viés
                m_hat = exp_avg.div(1 - beta1 ** t)
                v_hat = exp_avg_sq.div(1 - beta2 ** t)

                # Decay separado do weight update
                p.mul_(1 - lr * wd)
                # theta -= lr * m_hat / (sqrt(v_hat) + eps)
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

        return loss