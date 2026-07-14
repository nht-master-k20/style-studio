class FusionController:
    """Decide when to stop teacher attention fusion, based on TEACHER layout stability.

    Each self-attn hijack processor register()s at setup and report()s a per-step
    divergence d at every fusion step. The signal is the teacher self-attention's
    temporal change on the cond branch, d(t) = mean|A_teacher(t) - A_teacher(t-1)|:
    while the layout is still forming the teacher map changes a lot; once the layout is
    "locked" it barely changes, so continuing to force it onto the student adds nothing.
    (No previous map exists at the first fused step, so that step is not reported.)
    Once all registered layers have reported for step t, the controller computes r(t)
    and decides whether to stop fusing from step t+1 onward.

    Per-layer normalization uses a RUNNING MAX baseline so that layers with different
    attention scales/resolutions are comparable and a layer that keeps setting new highs
    stays at r_layer~1 until its change actually decays from the peak.
    """

    _EPS = 1e-12

    def __init__(self, rho=0.2, end_fusion_max=30, min_steps=5):
        self.rho = rho
        self.end_fusion_max = end_fusion_max
        self.min_steps = min_steps
        self.layers = []
        self.reset()

    def register(self, layer_name):
        if layer_name not in self.layers:
            self.layers.append(layer_name)

    def reset(self):
        self.stopped = False
        self.stop_step = None
        self._baseline = {}   # layer_name -> running max of d
        self._pending = {}    # step -> {layer_name: d}
        self.history = []     # [(step, r)]

    def is_active(self, step):
        return (not self.stopped) and step <= self.end_fusion_max

    def report(self, layer_name, step, d):
        if self.stopped:
            return
        self._pending.setdefault(step, {})[layer_name] = d
        if len(self._pending[step]) == len(self.layers):
            self._decide(step)

    def _decide(self, step):
        ratios = []
        for name, d in self._pending[step].items():
            base = max(self._baseline.get(name, 0.0), d)
            self._baseline[name] = base
            ratios.append(d / base if base > self._EPS else 0.0)
        r = sum(ratios) / len(ratios)
        self.history.append((step, r))
        del self._pending[step]
        if step >= self.min_steps and r <= self.rho:
            self.stopped = True
            self.stop_step = step

    def to_dict(self):
        return {
            "rho": self.rho,
            "end_fusion_max": self.end_fusion_max,
            "min_steps": self.min_steps,
            "stop_step": self.stop_step,
            "r_history": [[s, round(r, 6)] for s, r in self.history],
        }
