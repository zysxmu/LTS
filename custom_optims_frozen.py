import torch
from torch.optim.optimizer import Optimizer, required
import math
from torch import Tensor
from typing import List, Optional

__all__ = ['quant_SGD_frozen']

# ref. https://github.com/pytorch/pytorch/blob/1e64c8a8e37aceb82675b77afd71fd8fc78cd0bb/torch/optim/sgd.py
class quant_SGD_frozen(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize=False, foreach: Optional[bool] = None, 
                 is_quantized=False, initial_learning_rate=1e-1, fixed=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if is_quantized and lr != 1:
            raise ValueError("lr for quantized weights represent a scaling factor, and should be initially set to 1: {}".format(lr))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize, foreach=foreach,
                        is_quantized=is_quantized, initial_learning_rate=initial_learning_rate,fixed=fixed)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super(quant_SGD_frozen, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if not group['is_quantized']: # for fp weights
                params_with_grad = []
                d_p_list = []
                momentum_buffer_list = []
                has_sparse_grad = False

                for p in group['params']:
                    if p.grad is not None:
                        params_with_grad.append(p)
                        d_p_list.append(p.grad)
                        if p.grad.is_sparse:
                            has_sparse_grad = True

                        state = self.state[p]
                        if 'momentum_buffer' not in state:
                            momentum_buffer_list.append(None)
                        else:
                            momentum_buffer_list.append(state['momentum_buffer'])

                sgd(params_with_grad,
                    d_p_list,
                    momentum_buffer_list,
                    weight_decay=group['weight_decay'],
                    momentum=group['momentum'],
                    lr=group['lr'],
                    dampening=group['dampening'],
                    nesterov=group['nesterov'],
                    maximize=group['maximize'],
                    has_sparse_grad=has_sparse_grad,
                    foreach=group['foreach'])

                # update momentum_buffers in state
                for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                    state = self.state[p]
                    state['momentum_buffer'] = momentum_buffer
            else: # for latent weights to quantize
                params_with_grad = []
                d_p_list = []
                momentum_buffer_list = []
                has_sparse_grad = False
                state_steps = [] # number of iteration steps


                for p in group['params']:
                    if p.grad is not None:
                        params_with_grad.append(p)
                        d_p_list.append(p.grad)
                        if p.grad.is_sparse:
                            has_sparse_grad = True

                        state = self.state[p]
                        if len(state) == 0:
                            state['step'] = 0
                        if 'momentum_buffer' not in state:
                            momentum_buffer_list.append(None)
                        else:
                            momentum_buffer_list.append(state['momentum_buffer'])
                        state['step'] += 1
                        state_steps.append(state['step'])

                quant_sgd(params_with_grad,
                          d_p_list,
                          momentum_buffer_list,
                          weight_decay=group['weight_decay'],
                          momentum=group['momentum'],
                          lr=group['lr'],
                          dampening=group['dampening'],
                          nesterov=group['nesterov'],
                          maximize=group['maximize'],
                          has_sparse_grad=has_sparse_grad,
                          foreach=group['foreach'],
                          state_steps=state_steps,
                          initial_learning_rate=group['initial_learning_rate'],
                          fixed=group['fixed'])

                # update momentum_buffers in state
                for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                    state = self.state[p]
                    state['momentum_buffer'] = momentum_buffer

        return loss

def quant_sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        has_sparse_grad: bool = None,
        foreach: bool = None,
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool,
        state_steps: List[Optional[Tensor]],
              initial_learning_rate: float,
              fixed: bool = None):
    r"""Functional API that performs SGD algorithm computation.
    See :class:`~torch.optim.SGD` for details.
    """

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        raise NotImplementedError
        # func = _multi_tensor_sgd
    else:
        func = _single_tensor_quant_sgd

    func(params,
         d_p_list,
         momentum_buffer_list,
         weight_decay=weight_decay,
         momentum=momentum,
         lr=lr,
         dampening=dampening,
         nesterov=nesterov,
         has_sparse_grad=has_sparse_grad,
         maximize=maximize,
         state_steps=state_steps,
         initial_learning_rate=initial_learning_rate,
         fixed=fixed)

def _single_tensor_quant_sgd(params: List[Tensor],
                       d_p_list: List[Tensor],
                       momentum_buffer_list: List[Optional[Tensor]],
                       *,
                       weight_decay: float,
                       momentum: float,
                       lr: float,
                       dampening: float,
                       nesterov: bool,
                       maximize: bool,
                       has_sparse_grad: bool,
                       state_steps: List[Optional[Tensor]],
                             initial_learning_rate: float,
                             fixed: bool = None):

    for i, param in enumerate(params):

        d_p = d_p_list[i]
        step = state_steps[i]
        
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        ############ new update rule ############
        if step==1:
            if not hasattr(param, 'transition'):
                raise RuntimeError("The number of transitions is not counted")
            if maximize:
                alpha = 1.0
            else:
                alpha = -1.0
            param.add_(d_p, alpha=alpha)
        else:

            if fixed:
                d_p = d_p * initial_learning_rate * lr * (param.unchange_step == 0)
            else:
                d_p = d_p * initial_learning_rate * lr
            # d_p = d_p * TALR
            if maximize:
                alpha = 1.0
            else:
                alpha = -1.0
            param.add_(d_p, alpha=alpha)



def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
        # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
        has_sparse_grad: bool = None,
        foreach: bool = None,
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool):
    r"""Functional API that performs SGD algorithm computation.
    See :class:`~torch.optim.SGD` for details.
    """

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        raise NotImplementedError
        # func = _multi_tensor_sgd
    else:
        func = _single_tensor_sgd

    func(params,
         d_p_list,
         momentum_buffer_list,
         weight_decay=weight_decay,
         momentum=momentum,
         lr=lr,
         dampening=dampening,
         nesterov=nesterov,
         has_sparse_grad=has_sparse_grad,
         maximize=maximize)

def _single_tensor_sgd(params: List[Tensor],
                       d_p_list: List[Tensor],
                       momentum_buffer_list: List[Optional[Tensor]],
                       *,
                       weight_decay: float,
                       momentum: float,
                       lr: float,
                       dampening: float,
                       nesterov: bool,
                       maximize: bool,
                       has_sparse_grad: bool):

    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        alpha = lr if maximize else -lr
        param.add_(d_p, alpha=alpha)

