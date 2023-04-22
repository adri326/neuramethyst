use super::*;
use crate::layer::NeuraTrainableLayerBackprop;

impl<Input, Layer: NeuraLayer<Input>, ChildNetwork: NeuraLayer<Layer::Output>> NeuraLayer<Input>
    for NeuraSequential<Layer, ChildNetwork>
{
    type Output = ChildNetwork::Output;

    fn eval(&self, input: &Input) -> Self::Output {
        self.child_network.eval(&self.layer.eval(input))
    }
}

impl<
        Input,
        Layer: NeuraTrainableLayerBase<Input>,
        ChildNetwork: NeuraTrainableLayerBase<Layer::Output>,
    > NeuraTrainableLayerBase<Input> for NeuraSequential<Layer, ChildNetwork>
{
    type Gradient = (Layer::Gradient, Box<ChildNetwork::Gradient>);
    type IntermediaryRepr = (Layer::IntermediaryRepr, Box<ChildNetwork::IntermediaryRepr>);

    fn default_gradient(&self) -> Self::Gradient {
        (
            self.layer.default_gradient(),
            Box::new(self.child_network.default_gradient()),
        )
    }

    fn eval_training(&self, input: &Input) -> (Self::Output, Self::IntermediaryRepr) {
        let (layer_output, layer_intermediary) = self.layer.eval_training(input);
        let (child_output, child_intermediary) = self.child_network.eval_training(&layer_output);

        (
            child_output,
            (layer_intermediary, Box::new(child_intermediary)),
        )
    }

    fn prepare_layer(&mut self, is_training: bool) {
        self.layer.prepare_layer(is_training);
        self.child_network.prepare_layer(is_training);
    }

    fn apply_gradient(&mut self, gradient: &Self::Gradient) {
        self.layer.apply_gradient(&gradient.0);
        self.child_network.apply_gradient(&gradient.1);
    }
}

impl<
        Input,
        Layer: NeuraTrainableLayerSelf<Input>,
        ChildNetwork: NeuraTrainableLayerSelf<Layer::Output> + NeuraTrainableLayerBackprop<Layer::Output>,
    > NeuraTrainableLayerSelf<Input> for NeuraSequential<Layer, ChildNetwork>
{
    fn regularize_layer(&self) -> Self::Gradient {
        (
            self.layer.regularize_layer(),
            Box::new(self.child_network.regularize_layer()),
        )
    }

    fn get_gradient(
        &self,
        _input: &Input,
        _intermediary: &Self::IntermediaryRepr,
        _epsilon: &Self::Output,
    ) -> Self::Gradient {
        unimplemented!("NeuraSequential::get_gradient is not yet implemented, sorry");
    }
}

impl<
        Input,
        Layer: NeuraTrainableLayerBackprop<Input>,
        ChildNetwork: NeuraTrainableLayerBackprop<Layer::Output>,
    > NeuraTrainableLayerBackprop<Input> for NeuraSequential<Layer, ChildNetwork>
{
    fn backprop_layer(
        &self,
        input: &Input,
        intermediary: &Self::IntermediaryRepr,
        incoming_epsilon: &Self::Output,
    ) -> Input {
        let transient_output = self.layer.eval(input);
        let transient_epsilon =
            self.child_network
                .backprop_layer(&transient_output, &intermediary.1, incoming_epsilon);
        let outgoing_epsilon =
            self.layer
                .backprop_layer(input, &intermediary.0, &transient_epsilon);

        outgoing_epsilon
    }
}
