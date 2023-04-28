//! Implementations for NeuraLayer*
use std::borrow::Cow;

use crate::network::*;

use super::*;

impl<Axis, Layer, ChildNetwork> NeuraResidualNode<Layer, ChildNetwork, Axis> {
    fn process_input<Data>(
        &self,
        input: &NeuraResidualInput<Data>,
    ) -> (Axis::Combined, NeuraResidualInput<Data>)
    where
        Axis: NeuraCombineInputs<Data>,
        Layer: NeuraLayer<Axis::Combined>,
    {
        let (inputs, rest) = input.shift();

        let layer_input = self.axis.combine(inputs);

        (layer_input, rest)
    }

    fn combine_outputs<Data>(
        &self,
        layer_output: Data,
        output: &mut NeuraResidualInput<Data>,
    ) -> Rc<Data> {
        let layer_output = Rc::new(layer_output);

        for &offset in &self.offsets {
            output.push(offset, Rc::clone(&layer_output));
        }

        layer_output
    }

    pub(crate) fn map_input_owned<Data>(&self, input: &NeuraResidualInput<Data>) -> Axis::Combined
    where
        Axis: NeuraCombineInputs<Data>,
    {
        self.axis.combine(input.shift().0)
    }
}

impl<F: Float + Scalar, Layer, ChildNetwork, Axis> NeuraLayer<NeuraResidualInput<DVector<F>>>
    for NeuraResidualNode<Layer, ChildNetwork, Axis>
where
    Axis: NeuraCombineInputs<DVector<F>>,
    Layer: NeuraLayer<Axis::Combined, Output = DVector<F>>,
    ChildNetwork: NeuraLayer<NeuraResidualInput<DVector<F>>>,
{
    type Output = <ChildNetwork as NeuraLayer<NeuraResidualInput<DVector<F>>>>::Output;

    fn eval(&self, input: &NeuraResidualInput<DVector<F>>) -> Self::Output {
        let (layer_input, mut rest) = self.process_input(input);

        self.combine_outputs(self.layer.eval(&layer_input), &mut rest);

        self.child_network.eval(&rest)
    }
}

#[allow(dead_code)]
pub struct NeuraResidualIntermediary<LayerIntermediary, LayerOutput, ChildIntermediary> {
    layer_intermediary: LayerIntermediary,
    layer_output: Rc<LayerOutput>,
    child_intermediary: Box<ChildIntermediary>,
}

impl<Layer: NeuraTrainableLayerBase, ChildNetwork: NeuraTrainableLayerBase, Axis>
    NeuraTrainableLayerBase for NeuraResidualNode<Layer, ChildNetwork, Axis>
{
    type Gradient = (Layer::Gradient, Box<ChildNetwork::Gradient>);

    fn default_gradient(&self) -> Self::Gradient {
        (
            self.layer.default_gradient(),
            Box::new(self.child_network.default_gradient()),
        )
    }

    fn apply_gradient(&mut self, gradient: &Self::Gradient) {
        self.layer.apply_gradient(&gradient.0);
        self.child_network.apply_gradient(&gradient.1);
    }

    fn prepare_layer(&mut self, is_training: bool) {
        self.layer.prepare_layer(is_training);
        self.child_network.prepare_layer(is_training);
    }
}

impl<
        Data,
        Axis: NeuraCombineInputs<Data>,
        Layer: NeuraTrainableLayerEval<Axis::Combined, Output = Data>,
        ChildNetwork: NeuraTrainableLayerEval<NeuraResidualInput<Data>>,
    > NeuraTrainableLayerEval<NeuraResidualInput<Data>>
    for NeuraResidualNode<Layer, ChildNetwork, Axis>
where
    NeuraResidualNode<Layer, ChildNetwork, Axis>:
        NeuraLayer<NeuraResidualInput<Data>, Output = ChildNetwork::Output>,
{
    type IntermediaryRepr = NeuraResidualIntermediary<
        Layer::IntermediaryRepr,
        Layer::Output,
        ChildNetwork::IntermediaryRepr,
    >;

    fn eval_training(
        &self,
        input: &NeuraResidualInput<Data>,
    ) -> (Self::Output, Self::IntermediaryRepr) {
        let (layer_input, mut rest) = self.process_input(input);

        let (layer_output, layer_intermediary) = self.layer.eval_training(&layer_input);
        let layer_output = self.combine_outputs(layer_output, &mut rest);

        let (output, child_intermediary) = self.child_network.eval_training(&rest);

        let intermediary = NeuraResidualIntermediary {
            layer_intermediary,
            layer_output,
            child_intermediary: Box::new(child_intermediary),
        };

        (output, intermediary)
    }
}

impl<
        Data,
        Axis: NeuraCombineInputs<Data>,
        Layer: NeuraTrainableLayerSelf<Axis::Combined, Output = Data>,
        ChildNetwork: NeuraTrainableLayerSelf<NeuraResidualInput<Data>>,
    > NeuraTrainableLayerSelf<NeuraResidualInput<Data>>
    for NeuraResidualNode<Layer, ChildNetwork, Axis>
where
    NeuraResidualNode<Layer, ChildNetwork, Axis>:
        NeuraLayer<NeuraResidualInput<Data>, Output = ChildNetwork::Output>,
{
    fn regularize_layer(&self) -> Self::Gradient {
        (
            self.layer.regularize_layer(),
            Box::new(self.child_network.regularize_layer()),
        )
    }

    fn get_gradient(
        &self,
        input: &NeuraResidualInput<Data>,
        intermediary: &Self::IntermediaryRepr,
        epsilon: &Self::Output,
    ) -> Self::Gradient {
        unimplemented!();
    }
}

impl<Axis, Layer, ChildNetwork> NeuraNetworkBase for NeuraResidualNode<Layer, ChildNetwork, Axis> {
    type Layer = Layer;

    fn get_layer(&self) -> &Self::Layer {
        &self.layer
    }
}

impl<Axis, Layer: NeuraTrainableLayerBase, ChildNetwork: NeuraTrainableLayerBase> NeuraNetworkRec
    for NeuraResidualNode<Layer, ChildNetwork, Axis>
{
    type NextNode = ChildNetwork;

    fn get_next(&self) -> &Self::NextNode {
        &self.child_network
    }

    fn merge_gradient(
        &self,
        rec_gradient: <Self::NextNode as NeuraTrainableLayerBase>::Gradient,
        layer_gradient: <Self::Layer as NeuraTrainableLayerBase>::Gradient,
    ) -> Self::Gradient {
        (layer_gradient, Box::new(rec_gradient))
    }
}

impl<
        Data: Clone,
        Axis: NeuraCombineInputs<Data>,
        Layer: NeuraLayer<Axis::Combined, Output = Data>,
        ChildNetwork,
    > NeuraNetwork<NeuraResidualInput<Data>> for NeuraResidualNode<Layer, ChildNetwork, Axis>
where
    Layer::Output: Clone,
    Axis::Combined: Clone,
{
    type LayerInput = Axis::Combined;

    type NodeOutput = NeuraResidualInput<Data>;

    fn map_input<'a>(&'_ self, input: &'a NeuraResidualInput<Data>) -> Cow<'a, Self::LayerInput> {
        Cow::Owned(self.map_input_owned(input))
    }

    fn map_output<'a>(
        &'_ self,
        input: &'_ NeuraResidualInput<Data>,
        layer_output: &'a <Self::Layer as NeuraLayer<Self::LayerInput>>::Output,
    ) -> Cow<'a, Self::NodeOutput> {
        let mut remaining_inputs = input.shift().1;
        self.combine_outputs(layer_output.clone(), &mut remaining_inputs);

        Cow::Owned(remaining_inputs)
    }

    #[allow(unused_variables)]
    fn map_gradient_in<'a>(
        &'_ self,
        input: &'_ NeuraResidualInput<Data>,
        gradient_in: &'a Self::NodeOutput,
    ) -> Cow<'a, <Self::Layer as NeuraLayer<Self::LayerInput>>::Output> {
        // To convert from gradient_in to layer's gradient_in:
        // Pop the first value from `epsilon`, then:
        // - compute its sum
        // - use it to compute the outcoming epsilon of the current layer
        // - split the oucoming epsilon into its original components, and push those back onto the rest
        // At this point, the value for `epsilon` in the gradient solver's state should be ready for another iteration,
        // with the first value containing the unsummed incoming epsilon values from the downstream layers
        unimplemented!()
    }

    #[allow(unused_variables)]
    fn map_gradient_out<'a>(
        &'_ self,
        input: &'_ NeuraResidualInput<Data>,
        gradient_in: &'_ Self::NodeOutput,
        gradient_out: &'a Self::LayerInput,
    ) -> Cow<'a, NeuraResidualInput<Data>> {
        unimplemented!()
    }
}
