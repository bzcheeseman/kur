#
# Created by Aman LaChapelle on 3/15/17.
#
# pytorch-NTM
# Copyright (c) 2017 Aman LaChapelle
# Full license at pytorch-NTM/LICENSE.txt
#

import torch
import torch.nn as nn
from torch.autograd import Variable

from .head import *

# TODO: Worried about the forward function...
# TODO: Add CUDA support, and memory-saving option for small GPUs (added cuda support to the read/write heads)
# TODO: Benchmarking RNN needs to work with arbitrary sequence lengths.


class NTM(nn.Module):
    def __init__(self,
                 control,
                 read_head,
                 write_head,
                 memory_dims,
                 batch_size,
                 num_outputs):
        super(NTM, self).__init__()

        self.memory_dims = memory_dims
        self.memory = Variable(torch.rand(memory_dims[0], memory_dims[1]))
        self.controller = control
        self.read_head = read_head
        self.write_head = write_head
        self.batch_size = batch_size

        self.wr = Variable(torch.eye(self.batch_size, self.memory_dims[0]))
        self.ww = Variable(torch.eye(self.batch_size, self.memory_dims[0]))

        self.mem_bn = nn.BatchNorm1d(self.memory_dims[1])

        self.hid_to_out = nn.Linear(self.controller.num_hidden, num_outputs)

        self.hidden = Variable(
            torch.FloatTensor(batch_size, 1, self.controller.num_hidden)
                .normal_(0.0,  1. / self.controller.num_hidden))

    def get_weights_mem(self):
        return self.memory.cpu(), self.ww.cpu(), self.wr.cpu()

    def step(self, x_t):
        m_t = self.write_head(self.hidden, self.ww, self.memory, get_weights=False)  # write to memory

        r_t = self.read_head(self.hidden, self.wr, m_t, get_weights=False)  # read from memory

        h_t = self.controller(x_t, r_t)  # stores h_t in self.controller.hidden

        # the weights are getting corrupted here - they all end up the same...
        # get weights for next time around
        ww_t = self.write_head(h_t, self.ww, m_t, get_weights=True)
        wr_t = self.read_head(h_t, self.wr, m_t, get_weights=True)

        # update
        self.memory = self.mem_bn(m_t)
        # self.memory = m_t
        self.ww = ww_t
        self.wr = wr_t
        self.hidden = h_t

        out = Funct.sigmoid(self.hid_to_out(h_t))
        return out

    def forward(self, x):

        self.wr = Variable(torch.eye(self.batch_size, self.memory_dims[0]))
        self.ww = Variable(torch.eye(self.batch_size, self.memory_dims[0]))

        x = x.permute(1, 0, 2, 3)  # (time_steps, batch_size, features_rows, features_cols)

        outs = torch.stack(
            tuple(self.step(x_t) for x_t in torch.unbind(x, 0)),
            0)  # equvalent of Theano scan

        outs = outs.permute(1, 0, 2)  # (batch_size, time_steps, features)

        self.hidden = Variable(self.hidden.data)
        self.memory = Variable(self.memory.data)
        self.ww = Variable(self.ww.data)
        self.wr = Variable(self.wr.data)

        return outs


class BidirectionalNTM(nn.Module):
    def __init__(self,
                 control,
                 read_head,
                 write_head,
                 memory_dims,
                 batch_size,
                 num_outputs  # per time sequence
                 ):
        super(BidirectionalNTM, self).__init__()

        self.memory_dims = memory_dims
        self.memory = Variable(torch.FloatTensor(memory_dims[0], memory_dims[1]).fill_(5e-6))
        self.controller = control
        self.read_head = read_head
        self.write_head = write_head
        self.batch_size = batch_size

        self.wr = Variable(torch.eye(self.batch_size, self.memory_dims[0]))
        self.ww = Variable(torch.eye(self.batch_size, self.memory_dims[0]))

        # not sure about the num_outputs thing, thought it might be neat to see :P
        self.hid_to_out = nn.Linear(self.controller.num_hidden, int(num_outputs/2))

        self.mem_bn = nn.BatchNorm1d(self.memory_dims[1])

        self.hidden_fwd = Variable(torch.FloatTensor(batch_size, 1,
                                        self.controller.num_hidden)
                                   .normal_(0.0, 1. / self.controller.num_hidden))
        self.hidden_bwd = Variable(torch.FloatTensor(batch_size, 1,
                                        self.controller.num_hidden)
                                   .normal_(0.0, 1. / self.controller.num_hidden))

    def step_fwd(self, x_t):
        m_t = self.write_head(self.hidden_fwd, self.ww, self.memory, get_weights=False)  # write to memory

        r_t = self.read_head(self.hidden_fwd, self.wr, m_t, get_weights=False)  # read from memory

        h_t = self.controller(x_t, r_t)  # stores h_t in self.controller.hidden

        # the weights are getting corrupted here - they all end up the same
        # get weights for next time around
        ww_t = self.write_head(h_t, self.ww, m_t, get_weights=True)
        wr_t = self.read_head(h_t, self.wr, m_t, get_weights=True)

        # update
        self.memory = self.mem_bn(m_t)
        self.ww = ww_t
        self.wr = wr_t
        self.hidden_fwd = h_t

        out = Funct.sigmoid(self.hid_to_out(h_t))
        return out

    def step_bwd(self, x_t):
        m_t = self.write_head(self.hidden_bwd, self.ww, self.memory, get_weights=False)  # write to memory

        r_t = self.read_head(self.hidden_bwd, self.wr, m_t, get_weights=False)  # read from memory

        h_t = self.controller(x_t, r_t)  # stores h_t in self.controller.hidden

        # the weights are getting corrupted here - they all end up the same
        # get weights for next time around
        ww_t = self.write_head(h_t, self.ww, m_t, get_weights=True)
        wr_t = self.read_head(h_t, self.wr, m_t, get_weights=True)

        # update
        self.memory = self.mem_bn(m_t)
        self.ww = ww_t
        self.wr = wr_t
        self.hidden_fwd = h_t

        out = Funct.sigmoid(self.hid_to_out(h_t))
        return out

    def forward(self, x):
        self.wr = Variable(torch.eye(self.batch_size, self.memory_dims[0]))
        self.ww = Variable(torch.eye(self.batch_size, self.memory_dims[0]))

        x = x.permute(1, 0, 2, 3)  # (time_steps, batch_size, features_rows, features_cols)

        # forward pass
        outs_fwd = torch.stack(
            tuple(self.step_fwd(x_t) for x_t in torch.unbind(x, 0)),
            0)

        # backward pass
        outs_bwd = torch.stack(
            tuple(self.step_bwd(x_t) for x_t in reversed(torch.unbind(x, 0))),
            0)

        outs_fwd = outs_fwd.permute(1, 0, 2)  # (batch_size, time_steps, features)
        outs_bwd = outs_bwd.permute(1, 0, 2)
        outs = torch.stack((outs_fwd, outs_bwd), 1)

        self.hidden_fwd = Variable(self.hidden_fwd.data)
        self.hidden_bwd = Variable(self.hidden_bwd.data)
        self.memory = Variable(self.memory.data)
        self.ww = Variable(self.ww.data)
        self.wr = Variable(self.wr.data)

        return outs
