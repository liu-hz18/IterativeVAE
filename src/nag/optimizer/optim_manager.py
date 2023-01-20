from torch.nn.utils import clip_grad_norm_


class OptimizerManager(object):

    def __init__(self, model, optimizer, scheduler, update_freq=1, max_norm=100.0):
        super(OptimizerManager, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.update_freq = update_freq
        self.max_norm = max_norm
        self.demonitor = 0
        self.count = 0
        self.last_total_norm = 0.
        self.total_loss = 0.

    def backward(self, loss, demon):
        self.total_loss += loss.item()
        loss.backward()
        self.demonitor += demon
        self.count += 1
        return self.total_loss / self.demonitor

    def step(self):
        if (self.count+1) % self.update_freq == 0:
            self._multiply_grads(1. / self.demonitor)
            self.last_total_norm = clip_grad_norm_(
                self.model.parameters(), max_norm=self.max_norm, norm_type=2).item()
            self.optimizer.step()
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    p.grad = None
            self.optimizer.zero_grad()
            self.scheduler.step()
            self.demonitor = 0
            self.count = 0
            self.total_loss = 0.
        return self.last_total_norm

    def _multiply_grads(self, c):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(c)
