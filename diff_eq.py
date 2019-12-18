import time
import numpy as np
import torch
import warnings
from collections import deque

warnings.filterwarnings('ignore', message='To copy construct from a tensor')
torch.set_default_tensor_type(torch.DoubleTensor)


class SecondOrderDiffEq:
    """
    A class that evaluates 2nd order nonlinear diffential equations
    of 2 variables using a discrete finite difference
    """
    def __init__(self, diff_eq_fn, num_weights, weights=None, h=0.01):
        self.diff_eq_fn = diff_eq_fn
        self.h = h
        self.num_weights = num_weights

        if weights is None:
            weights = torch.randn(num_weights)

        assert self.num_weights == len(weights)
        self.weights = torch.tensor(weights)

    def evaluate(self, f):
        """
        Evaluate this DiffEq with the given values of the function value
        specified by `f`. `f` should be a 2D torch.tensor

        This uses a discrete finite difference.
        """
        height, width = f.shape

        # constant term
        constant = torch.ones((f.shape[0] - 2, f.shape[1] - 2))

        # x, y coordinates
        x, y = np.meshgrid(np.arange(1, width - 1), np.arange(1, height - 1))
        x = torch.tensor(x).double() * self.h
        y = torch.tensor(y).double() * self.h

        # f_x and f_y
        f_x = (f[1:-1, 2:] - f[1:-1, :-2]) / (2 * self.h)
        f_y = (f[2:, 1:-1] - f[:-2, 1:-1]) / (2 * self.h)

        # f_xx and f_yy
        f_xx = (f[1:-1, 2:] - 2*f[1:-1,1:-1] + f[1:-1, :-2]) / (self.h ** 2)
        f_yy = (f[2:, 1:-1] - 2*f[1:-1,1:-1] + f[:-2, 1:-1]) / (self.h ** 2)

        # f_xy
        f_xy = (f[2:, 2:] - f[:-2, 2:] - f[2:, :-2] + f[:-2, :-2]) / (4 * self.h ** 2)

        # the internal part of f that excludes the boundary conditions
        f_internal = f[1:-1, 1:-1]

        res_internal = self.diff_eq_fn(
            weights=self.weights,
            constant=constant, x=x, y=y, f=f_internal,
            f_x=f_x, f_y=f_y,
            f_xx=f_xx, f_xy=f_xy, f_yy=f_yy
        )

        return self._hardcode_boundary(torch.zeros(f.shape), res_internal)

    def fit(self, orig_image, lr=0.1, num_epochs=5000, weight_penalty=0, **opt_kwargs):
        """
        Fit a differential equation to an image
        """
        orig_image = torch.tensor(orig_image).double()
        self.weights.requires_grad = True

        def closure():
            loss = torch.mean(self.evaluate(orig_image) ** 2)
            loss += weight_penalty * torch.mean(1 / (self.weights.abs() + 1e-6))
            return loss

        print('\nFitting Diff Eq Weights\n=======================')
        self._lbfgs_training_loop([self.weights], closure, num_epochs, lr=lr, **opt_kwargs)
        self.weights.requires_grad = False
        return self

    def solve(self, orig_image, lr=0.1, num_epochs=5000, **opt_kwargs):
        """
        Solve this differential equation using the boundary conditions
        specified by `orig_image`. Only the boundaries of `orig_image`
        are used, the remaining part of it can be arbitrary data.
        """
        internal_shape = (orig_image.shape[0] - 2, orig_image.shape[1] - 2)
        f_internal = torch.randn(internal_shape, requires_grad=True)

        def closure():
            f = self._hardcode_boundary(orig_image, f_internal)
            loss = torch.mean(self.evaluate(f) ** 2)
            return loss

        print('\nSolving Diff Eq\n===============')
        self._lbfgs_training_loop([f_internal], closure, num_epochs, lr=lr, **opt_kwargs)
        return self._hardcode_boundary(orig_image, f_internal).detach()

    def _hardcode_boundary(self, f, internal_f):
        assert f[1:-1, 1:-1].shape == internal_f.shape

        res = f.clone()
        res[1:-1, 1:-1] = internal_f
        return res

    def _lbfgs_training_loop(self, parameters, closure, num_epochs=5000,
                             stagnation_period=100, **opt_kwargs):
        opt = torch.optim.LBFGS(parameters, **opt_kwargs)

        start_time = time.time()
        last_time = None
        history = deque(maxlen=stagnation_period)

        for i in range(num_epochs):
            try:
                opt.zero_grad()
                loss = closure()
                loss.backward()
                opt.step(lambda: loss)

                history.append(loss.detach())
                if len(history) == stagnation_period and min(history) / max(history) > 0.95:
                    print(f'loss has not improved in {stagnation_period} iterations. stopping')
                    break

                if last_time is None or time.time() - last_time > 1:
                    tot_time = time.time() - start_time
                    print(f'epoch {i} | {tot_time:.01f} sec | loss: {loss:.3g}')
                    last_time = time.time()

            except KeyboardInterrupt:
                print('stopping early')
                break

        tot_time = time.time() - start_time
        print(f'done | epoch {i} | {tot_time:.01f} sec | loss: {loss:.3g}\n')
