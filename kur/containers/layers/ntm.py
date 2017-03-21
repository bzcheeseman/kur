"""
Copyright 2016 Aman LaChapelle

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from . import Layer, ParsingError


###############################################################################
class NTM(Layer):
    """ A Neural Turing Machine

        # Properties

        memory_dims: (int, int).
        batch_size: int.
        num_inputs: int.
        num_hidden: int.
        num_outputs: int.
        bidirectional: bool.
        controller_type: one of (feed_forward)

        # Example

        ```
        ntm:
            memory_dims: (128, 20)
            batch_size: 1
            num_inputs: 8
            num_hidden: 100
            num_outputs: 8
            bidirectional: no
            controller_type: feed_forward
        ```

    """

    ###########################################################################
    def __init__(self, *args, **kwargs):

        """Creates a new Neural Turing Machine layer
        """

        super().__init__(*args, **kwargs)
        self.memory_dims = None  # memory_dims
        self.batch_size = None  # batch_size
        self.num_inputs = None
        self.num_hidden = None
        self.num_outputs = None
        self.bidirectional = None
        self.controller_type = None

        self.controller = None  # control
        self.read_head = None  # read_head
        self.write_head = None  # write_head

    ###########################################################################
    def _parse(self, engine):
        super()._parse(engine)

        if 'memory_dims' not in self.args:
            raise ParsingError('Missing key "memory_dims" in ntm container')
        self.memory_dims = engine.evaluate(self.args['memory_dims'])

        if 'batch_size' not in self.args:
            raise ParsingError('Missing key "batch_size" in ntm container')
        self.batch_size = engine.evaluate(self.args['batch_size'])

        if 'num_inputs' not in self.args:
            raise ParsingError('Missing key "num_inputs" in ntm container')
        self.num_inputs = engine.evaluate(self.args['num_inputs'])

        if 'num_hidden' not in self.args:
            raise ParsingError('Missing key "num_hidden" in ntm container')
        self.num_hidden = engine.evaluate(self.args['num_hidden'])

        if 'num_outputs' not in self.args:
            raise ParsingError('Missing key "num_outputs" in ntm container')
        self.num_outputs = engine.evaluate(self.args['num_outputs'])

        if 'bidirectional' not in self.args:
            raise ParsingError('Missing key "bidirectional" in ntm container')
        self.bidirectional = engine.evaluate(self.args['bidirectional'])

        if 'controller_type' not in self.args:
            raise ParsingError('Missing key "controller_type" in ntm container')
        self.controller_type = engine.evaluate(self.args['controller_type'])
        if self.controller_type != 'feed_forward':
            raise NotImplementedError("Only feed_forward controllers are implemented yet")

    ###########################################################################
    def _build(self, model):
        backend = model.get_backend()

        if backend.get_name() != 'pytorch':
            raise NotImplementedError("NTM not implemented for backend other than pytorch")

        from .ntm_backend import ReadHead, WriteHead, NTM, BidirectionalNTM, FeedForwardController

        self.controller = FeedForwardController(self.num_inputs, self.num_hidden, self.batch_size, 1, self.memory_dims)
        self.read_head = ReadHead(self.num_hidden)
        self.write_head = WriteHead(self.num_hidden)

        ntm = None

        if self.bidirectional:
            ntm = BidirectionalNTM
        else:
            ntm = NTM

        def connect(inputs):
            kwargs = {
                'control': self.controller,
                'read_head': self.read_head,
                'write_head': self.write_head,
                'memory_dims': self.memory_dims,
                'batch_size': self.batch_size,
                'num_outputs': self.num_outputs
            }

            def layer_func(layer, *inputs):
                result = layer(*inputs)
                return result

            return {
                'shape': self.shape([inputs[0]['shape']]),
                'layer': model.data.add_layer(
                    self.name,
                    ntm(**kwargs),
                    func=layer_func
                )(inputs[0]['layer'])
            }

        yield connect

    ###########################################################################
    def shape(self, input_shapes):
        if len(input_shapes) > 1:
            raise ValueError('Recurrent layers only take a single input.')
        input_shape = input_shapes[0]
        return input_shape[:-1] + (self.num_outputs, )



### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF
