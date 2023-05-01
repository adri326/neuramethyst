use std::borrow::Cow;

use crate::network::*;

use super::*;

#[derive(Clone, Debug, PartialEq)]
pub struct NeuraResidual<Layers> {
    /// Instance of NeuraResidualNode
    pub(crate) layers: Layers,

    /// Array of which layers to send the input to, defaults to `vec![0]`
    pub(crate) initial_offsets: Vec<usize>,
}

impl<Layers> NeuraResidual<Layers> {
    pub fn new(layers: Layers) -> Self {
        Self {
            layers,
            initial_offsets: vec![0],
        }
    }

    pub fn offset(mut self, offset: usize) -> Self {
        self.initial_offsets.push(offset);
        self
    }

    pub fn offsets(mut self, offsets: Vec<usize>) -> Self {
        self.initial_offsets = offsets;
        self
    }

    fn input_to_residual_input<Input: Clone>(&self, input: &Input) -> NeuraResidualInput<Input> {
        let input: Rc<Input> = Rc::new((*input).clone());
        let mut inputs = NeuraResidualInput::new();

        for &offset in &self.initial_offsets {
            inputs.push(offset, Rc::clone(&input));
        }

        drop(input);

        inputs
    }
}

impl<Input: Clone, Layers> NeuraLayer<Input> for NeuraResidual<Layers>
where
    Layers: NeuraLayer<NeuraResidualInput<Input>>,
{
    type Output = Layers::Output;

    fn eval(&self, input: &Input) -> Self::Output {
        self.layers.eval(&self.input_to_residual_input(input))
    }
}

impl<Layers: NeuraTrainableLayerBase> NeuraTrainableLayerBase for NeuraResidual<Layers> {
    type Gradient = Layers::Gradient;

    #[inline(always)]
    fn default_gradient(&self) -> Self::Gradient {
        self.layers.default_gradient()
    }

    #[inline(always)]
    fn apply_gradient(&mut self, gradient: &Self::Gradient) {
        self.layers.apply_gradient(gradient);
    }
}

impl<Data: Clone, Layers: NeuraTrainableLayerEval<NeuraResidualInput<Data>>>
    NeuraTrainableLayerEval<Data> for NeuraResidual<Layers>
{
    type IntermediaryRepr = Layers::IntermediaryRepr;

    fn eval_training(&self, input: &Data) -> (Self::Output, Self::IntermediaryRepr) {
        self.layers
            .eval_training(&self.input_to_residual_input(input))
    }
}

impl<Data: Clone, Layers: NeuraTrainableLayerSelf<NeuraResidualInput<Data>>>
    NeuraTrainableLayerSelf<Data> for NeuraResidual<Layers>
{
    fn regularize_layer(&self) -> Self::Gradient {
        self.layers.regularize_layer()
    }

    fn get_gradient(
        &self,
        input: &Data,
        intermediary: &Self::IntermediaryRepr,
        epsilon: &Self::Output,
    ) -> Self::Gradient {
        self.layers
            .get_gradient(&self.input_to_residual_input(input), intermediary, &epsilon)
    }
}

impl<Layers> NeuraNetworkBase for NeuraResidual<Layers> {
    type Layer = ();

    #[inline(always)]
    fn get_layer(&self) -> &Self::Layer {
        &()
    }
}

impl<Layers: NeuraTrainableLayerBase> NeuraNetworkRec for NeuraResidual<Layers> {
    type NextNode = Layers;

    #[inline(always)]
    fn get_next(&self) -> &Self::NextNode {
        &self.layers
    }

    #[inline(always)]
    fn merge_gradient(
        &self,
        rec_gradient: <Self::NextNode as NeuraTrainableLayerBase>::Gradient,
        _layer_gradient: <Self::Layer as NeuraTrainableLayerBase>::Gradient,
    ) -> Self::Gradient {
        rec_gradient
    }
}

impl<Data: Clone, Layers> NeuraNetwork<Data> for NeuraResidual<Layers> {
    type LayerInput = Data;
    type NodeOutput = NeuraResidualInput<Data>;

    #[inline(always)]
    fn map_input<'a>(&'_ self, input: &'a Data) -> Cow<'a, Self::LayerInput> {
        Cow::Borrowed(input)
    }

    #[inline(always)]
    fn map_output<'a>(
        &'_ self,
        _input: &'_ Data,
        layer_output: &'a Data,
    ) -> Cow<'a, Self::NodeOutput> {
        let layer_output = Rc::new(layer_output.clone());
        let mut outputs = NeuraResidualInput::new();

        for &offset in &self.initial_offsets {
            outputs.push(offset, Rc::clone(&layer_output));
        }

        Cow::Owned(outputs)
    }

    #[inline(always)]
    fn map_gradient_in<'a>(
        &'_ self,
        _input: &'_ Data,
        gradient_in: &'a Self::NodeOutput,
    ) -> Cow<'a, Data> {
        let first = gradient_in
            .clone()
            .get_first()
            .expect("No outgoing gradient in NeuraResidual on the last node");

        Cow::Owned((*first).clone())
    }

    #[inline(always)]
    fn map_gradient_out<'a>(
        &'_ self,
        _input: &'_ Data,
        _gradient_in: &'_ Self::NodeOutput,
        gradient_out: &'a Self::LayerInput,
    ) -> Cow<'a, Data> {
        Cow::Borrowed(gradient_out)
    }
}
