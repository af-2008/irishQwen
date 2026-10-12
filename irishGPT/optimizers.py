import torch


def find_parameters(obj, seen=None):
    """
    Walk a model and yield (layer, name) for every trainable tensor.

    We rely on the convention used by every layer in irishGPT: the gradient of
    parameter `X` is stored on the same layer as `dX` (W/dW, b/db, gamma/dgamma, ...).
    So a parameter is any tensor attribute that has a matching gradient attribute.
    """
    seen = seen if seen is not None else set()
    if id(obj) in seen:
        return
    seen.add(id(obj))

    if isinstance(obj, list):
        for item in obj:
            yield from find_parameters(item, seen)
        return

    if not hasattr(obj, "__dict__"):
        return

    for name, value in list(vars(obj).items()):
        if isinstance(value, torch.Tensor):
            if isinstance(getattr(obj, "d" + name, None), torch.Tensor):
                yield obj, name
        else:
            yield from find_parameters(value, seen)


class Optimizer:
    """
    Base class. Each step: find every (parameter, gradient) pair, let the subclass
    update the parameter, then delete the gradient so it can never be reused.
    (Matters for MoE: an expert that got no tokens this batch has no fresh
    gradient, so it gets skipped instead of re-applying last batch's.)
    """
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr
        self.state = {}     # per-parameter memory, keyed by (id(layer), name)

    def step(self):
        for layer, name in find_parameters(self.model):
            W = getattr(layer, name)
            dW = getattr(layer, "d" + name).reshape(W.shape)   # e.g. LinearLayer.db is (out,), b is (1, out)
            key = (id(layer), name)
            if key not in self.state:
                self.state[key] = {}

            self.update(W, dW, self.state[key])
            delattr(layer, "d" + name)

    def update(self, W, dW, state):
        raise NotImplementedError


class SGD(Optimizer):
    """
    Plain gradient descent -- exactly what every layer's update(lr) already does.
      W = W - lr * dW
    """
    def update(self, W, dW, state):
        W -= self.lr * dW


class Momentum(Optimizer):
    """
    SGD with momentum: keep a running "velocity" and step along it.
      v = beta * v + dW
      W = W - lr * v

    Like a ball rolling downhill: directions that keep agreeing batch after batch
    build up speed, directions that flip sign every batch (noise) cancel out.
    """
    def __init__(self, model, lr=0.1, beta=0.9):
        super().__init__(model, lr)
        self.beta = beta

    def update(self, W, dW, state):
        if "v" not in state:
            state["v"] = torch.zeros_like(W)

        state["v"] = self.beta * state["v"] + dW
        W -= self.lr * state["v"]


class Adam(Optimizer):
    """
    Adam = momentum + a separate step size for every single weight.
      m = beta1 * m + (1 - beta1) * dW          running average of the gradient   (direction)
      v = beta2 * v + (1 - beta2) * dW^2        running average of its square      (typical size)
      W = W - lr * m / (sqrt(v) + eps)

    Dividing by sqrt(v) normalizes every weight's step to roughly lr, whether its
    gradients are huge (embedding rows for common tokens) or tiny (rare tokens,
    deep layers). With plain SGD one lr has to suit all of them at once.

    m and v start at 0, so early on they're biased toward 0. Dividing by
    (1 - beta^t) undoes that ("bias correction").
    """
    def __init__(self, model, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(model, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def update(self, W, dW, state):
        if "m" not in state:
            state["m"] = torch.zeros_like(W)
            state["v"] = torch.zeros_like(W)
            state["t"] = 0

        state["t"] += 1
        state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * dW
        state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * dW * dW

        m_hat = state["m"] / (1 - self.beta1 ** state["t"])
        v_hat = state["v"] / (1 - self.beta2 ** state["t"])

        W -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    """
    Adam + weight decay: every step, also shrink each weight a little toward 0.
      W = W - lr * weight_decay * W         (then the usual Adam step)

    A weight only stays large if the gradient keeps pushing it there, so the
    model can't cheaply grow big weights just to memorize specific training text.

    The "W" (decoupled) part: the shrink is applied to W directly, NOT added
    to the gradient. If we added it to dW, Adam's divide-by-sqrt(v) would
    rescale it differently for every weight and weaken it unpredictably.

    We only decay matrices (embeddings, projections). Norm gammas are 1D
    scale factors whose "natural" value is 1, not 0, so they're left alone.
    """
    def __init__(self, model, lr=1e-3, weight_decay=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(model, lr, beta1, beta2, eps)
        self.weight_decay = weight_decay

    def update(self, W, dW, state):
        if W.dim() >= 2:
            W -= self.lr * self.weight_decay * W
        super().update(W, dW, state)
