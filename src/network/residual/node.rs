use std::borrow::Cow;

use crate::network::*;

use super::*;

#[derive(Clone, Debug, PartialEq)]
pub struct NeuraResidualNode<Layer, ChildNetwork, Axis> {
    pub layer: Layer,
    pub child_network: ChildNetwork,

    /// Array of relative layers indices to send the offset of this layer to,
    /// defaults to `vec![0]`.
    pub(crate) offsets: Vec<usize>,

    pub axis: Axis,

    pub(crate) output_shape: Option<NeuraShape>,
    pub(crate) input_shapes: Vec<NeuraShape>,
    pub(crate) input_offsets: Vec<usize>,
}

impl<Layer, ChildNetwork> NeuraResidualNode<Layer, ChildNetwork, NeuraAxisAppend> {
    pub fn new(layer: Layer, child_network: ChildNetwork) -> Self {
        Self {
            layer,
            child_network,
            offsets: vec![0],
            axis: NeuraAxisAppend,
            output_shape: None,
            input_shapes: vec![],
            input_offsets: vec![],
        }
    }
}

impl<Layer, ChildNetwork, Axis> NeuraResidualNode<Layer, ChildNetwork, Axis> {
    pub fn offsets(mut self, offsets: Vec<usize>) -> Self {
        self.offsets = offsets;
        self
    }

    pub fn offset(mut self, offset: usize) -> Self {
        self.offsets.push(offset);
        self
    }

    pub fn axis<Axis2>(self, axis: Axis2) -> NeuraResidualNode<Layer, ChildNetwork, Axis2> {
        if self.output_shape.is_some() {
            unimplemented!(
                "NeuraResidualNode::axis cannot yet be called after NeuraResidualNode::construct"
            );
        }

        NeuraResidualNode {
            layer: self.layer,
            child_network: self.child_network,
            offsets: self.offsets,
            axis,
            output_shape: None,
            input_shapes: self.input_shapes,
            input_offsets: self.input_offsets,
        }
    }

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

#[allow(dead_code)]
pub struct NeuraResidualIntermediary<LayerIntermediary, LayerOutput, ChildIntermediary> {
    layer_intermediary: LayerIntermediary,
    layer_output: Rc<LayerOutput>,
    child_intermediary: Box<ChildIntermediary>,
}

impl<
        Layer: NeuraLayerBase,
        ChildNetwork: NeuraLayerBase,
        Axis: Clone + std::fmt::Debug + 'static,
    > NeuraLayerBase for NeuraResidualNode<Layer, ChildNetwork, Axis>
{
    #[inline(always)]
    fn output_shape(&self) -> NeuraShape {
        self.child_network.output_shape()
    }

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

    fn regularize_layer(&self) -> Self::Gradient {
        (
            self.layer.regularize_layer(),
            Box::new(self.child_network.regularize_layer()),
        )
    }
}

impl<Data: Clone + 'static, Layer, ChildNetwork, Axis: Clone + std::fmt::Debug + 'static>
    NeuraLayer<NeuraResidualInput<Data>> for NeuraResidualNode<Layer, ChildNetwork, Axis>
where
    Axis: NeuraCombineInputs<Data>,
    Layer: NeuraLayer<Axis::Combined, Output = Data>,
    ChildNetwork: NeuraLayer<NeuraResidualInput<Data>>,
{
    type Output = <ChildNetwork as NeuraLayer<NeuraResidualInput<Data>>>::Output;
    type IntermediaryRepr = NeuraResidualIntermediary<
        Layer::IntermediaryRepr,
        Layer::Output,
        ChildNetwork::IntermediaryRepr,
    >;

    fn eval(&self, input: &NeuraResidualInput<Data>) -> Self::Output {
        let (layer_input, mut rest) = self.process_input(input);

        self.combine_outputs(self.layer.eval(&layer_input), &mut rest);

        self.child_network.eval(&rest)
    }

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

    #[allow(unused)]
    fn get_gradient(
        &self,
        input: &NeuraResidualInput<Data>,
        intermediary: &Self::IntermediaryRepr,
        epsilon: &Self::Output,
    ) -> Self::Gradient {
        unimplemented!("NeuraResidualNode::get_gradient is not yet implemented");
    }

    #[allow(unused)]
    fn backprop_layer(
        &self,
        input: &NeuraResidualInput<Data>,
        intermediary: &Self::IntermediaryRepr,
        epsilon: &Self::Output,
    ) -> NeuraResidualInput<Data> {
        unimplemented!("NeuraResidualNode::backprop_layer is not yet implemented");
    }
}

impl<Axis, Layer, ChildNetwork> NeuraNetworkBase for NeuraResidualNode<Layer, ChildNetwork, Axis> {
    type Layer = Layer;

    fn get_layer(&self) -> &Self::Layer {
        &self.layer
    }
}

impl<
        Axis: Clone + std::fmt::Debug + 'static,
        Layer: NeuraLayerBase,
        ChildNetwork: NeuraLayerBase,
    > NeuraNetworkRec for NeuraResidualNode<Layer, ChildNetwork, Axis>
{
    type NextNode = ChildNetwork;

    fn get_next(&self) -> &Self::NextNode {
        &self.child_network
    }

    fn merge_gradient(
        &self,
        rec_gradient: <Self::NextNode as NeuraLayerBase>::Gradient,
        layer_gradient: <Self::Layer as NeuraLayerBase>::Gradient,
    ) -> Self::Gradient {
        (layer_gradient, Box::new(rec_gradient))
    }
}

impl<
        Data: Clone + std::fmt::Debug,
        Axis: NeuraCombineInputs<Data> + NeuraSplitInputs<Data>,
        Layer: NeuraLayer<Axis::Combined, Output = Data> + std::fmt::Debug,
        ChildNetwork,
    > NeuraNetwork<NeuraResidualInput<Data>> for NeuraResidualNode<Layer, ChildNetwork, Axis>
where
    Layer::Output: Clone,
    Axis::Combined: Clone,
    for<'a> Data: std::iter::Sum<&'a Data>,
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

    // To convert from gradient_in to gradient_out:
    // - pop the first value from `gradient_in` (map_gradient_in)
    // - compute its sum (map_gradient_in)
    // - use it to compute the outcoming epsilon of the current layer (backprop)
    // - split the oucoming epsilon into its original components (map_gradient_out)
    // - push those back onto the rest (map_gradient_out)
    // At this point, the value for `epsilon` in the gradient solver's state should be ready for another iteration,
    // with the first value containing the unsummed incoming epsilon values from the downstream layers

    #[allow(unused_variables)]
    fn map_gradient_in<'a>(
        &'_ self,
        input: &'_ NeuraResidualInput<Data>,
        gradient_in: &'a Self::NodeOutput,
    ) -> Cow<'a, <Self::Layer as NeuraLayer<Self::LayerInput>>::Output> {
        let (first_gradient, _) = gradient_in.shift();

        let sum = first_gradient.iter().map(|x| x.as_ref()).sum();

        Cow::Owned(sum)
    }

    #[allow(unused_variables)]
    fn map_gradient_out<'a>(
        &'_ self,
        input: &'_ NeuraResidualInput<Data>,
        gradient_in: &'_ Self::NodeOutput,
        gradient_out: &'a Self::LayerInput,
    ) -> Cow<'a, NeuraResidualInput<Data>> {
        let (_, mut rest) = gradient_in.shift();

        let split = self.axis.split(gradient_out, &self.input_shapes);

        for (offset, gradient) in self.input_offsets.iter().copied().zip(split.into_iter()) {
            rest.push(offset, Rc::new(gradient));
        }

        Cow::Owned(rest)
    }
}
