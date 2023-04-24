//! Implementations for NeuraLayer*

use crate::{gradient_solver::NeuraGradientSolverTransient, network::{NeuraTrainableNetwork, NeuraTrainableNetworkBase}};

use super::*;

impl<Axis, Layer, ChildNetwork> NeuraResidualNode<Layer, ChildNetwork, Axis> {
    fn process_input<Data>(&self, input: &NeuraResidualInput<Data>) -> (Axis::Combined, NeuraResidualInput<Data>)
    where
        Axis: NeuraCombineInputs<Data>,
        Layer: NeuraLayer<Axis::Combined>
    {
        let (inputs, rest) = input.shift();

        let layer_input = self.axis.combine(inputs);

        (layer_input, rest)
    }

    fn combine_outputs<Data>(&self, layer_output: Data, output: &mut NeuraResidualInput<Data>) -> Rc<Data> {
        let layer_output = Rc::new(layer_output);

        for &offset in &self.offsets {
            output.push(offset, Rc::clone(&layer_output));
        }

        layer_output
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

impl<F: Clone, Output: Clone, Layers> NeuraLayer<DVector<F>> for NeuraResidual<Layers>
where
    Layers: NeuraLayer<NeuraResidualInput<DVector<F>>, Output = NeuraResidualInput<Output>>,
{
    type Output = Output;

    fn eval(&self, input: &DVector<F>) -> Self::Output {
        let input: Rc<DVector<F>> = Rc::new((*input).clone());
        let mut inputs = NeuraResidualInput::new();

        for &offset in &self.initial_offsets {
            inputs.push(offset, Rc::clone(&input));
        }

        drop(input);

        let output = self.layers.eval(&inputs);

        let result = output.get_first()
            .expect("Invalid NeuraResidual state: network returned no data, did you forget to link the last layer?")
            .into();

        Rc::unwrap_or_clone(result)
    }
}

pub struct NeuraResidualIntermediary<LayerIntermediary, LayerOutput, ChildIntermediary> {
    layer_intermediary: LayerIntermediary,
    layer_output: Rc<LayerOutput>,
    child_intermediary: Box<ChildIntermediary>,
}

impl<
    Data,
    Axis: NeuraCombineInputs<Data>,
    Layer: NeuraTrainableLayerBase<Axis::Combined, Output = Data>,
    ChildNetwork: NeuraTrainableLayerBase<NeuraResidualInput<Data>>
> NeuraTrainableLayerBase<NeuraResidualInput<Data>> for NeuraResidualNode<Layer, ChildNetwork, Axis>
where
    NeuraResidualNode<Layer, ChildNetwork, Axis>: NeuraLayer<NeuraResidualInput<Data>, Output=ChildNetwork::Output>
{
    type Gradient = (Layer::Gradient, Box<ChildNetwork::Gradient>);
    type IntermediaryRepr = NeuraResidualIntermediary<Layer::IntermediaryRepr, Layer::Output, ChildNetwork::IntermediaryRepr>;

    fn default_gradient(&self) -> Self::Gradient {
        (self.layer.default_gradient(), Box::new(self.child_network.default_gradient()))
    }

    fn apply_gradient(&mut self, gradient: &Self::Gradient) {
        self.layer.apply_gradient(&gradient.0);
        self.child_network.apply_gradient(&gradient.1);
    }

    fn eval_training(&self, input: &NeuraResidualInput<Data>) -> (Self::Output, Self::IntermediaryRepr) {
        let (layer_input, mut rest) = self.process_input(input);

        let (layer_output, layer_intermediary) = self.layer.eval_training(&layer_input);
        let layer_output = self.combine_outputs(layer_output, &mut rest);

        let (output, child_intermediary) = self.child_network.eval_training(&rest);

        let intermediary = NeuraResidualIntermediary {
            layer_intermediary,
            layer_output,
            child_intermediary: Box::new(child_intermediary)
        };

        (output, intermediary)
    }

    fn prepare_layer(&mut self, is_training: bool) {
        self.layer.prepare_layer(is_training);
        self.child_network.prepare_layer(is_training);
    }
}

impl<
    Data,
    Axis: NeuraCombineInputs<Data>,
    Layer: NeuraTrainableLayerSelf<Axis::Combined, Output = Data>,
    ChildNetwork: NeuraTrainableNetworkBase<NeuraResidualInput<Data>>,
> NeuraTrainableNetworkBase<NeuraResidualInput<Data>> for NeuraResidualNode<Layer, ChildNetwork, Axis>
where
    Self: NeuraTrainableLayerBase<NeuraResidualInput<Data>, Gradient = (Layer::Gradient, Box<ChildNetwork::Gradient>)>,
{
    type Gradient = <Self as NeuraTrainableLayerBase<NeuraResidualInput<Data>>>::Gradient;
    type LayerOutput = Layer::Output;

    fn default_gradient(&self) -> Self::Gradient {
        <Self as NeuraTrainableLayerBase<NeuraResidualInput<Data>>>::default_gradient(self)
    }

    fn apply_gradient(&mut self, gradient: &Self::Gradient) {
        <Self as NeuraTrainableLayerBase<NeuraResidualInput<Data>>>::apply_gradient(self, gradient)
    }

    fn regularize(&self) -> Self::Gradient {
        (self.layer.regularize_layer(), Box::new(self.child_network.regularize()))
    }

    fn prepare(&mut self, train_iteration: bool) {
        self.layer.prepare_layer(train_iteration);
        self.child_network.prepare(train_iteration);
    }
}

impl<
    Data,
    Axis: NeuraSplitInputs<Data>,
    Layer: NeuraTrainableLayerSelf<Axis::Combined, Output = Data>,
    Optimizer: NeuraGradientSolverTransient<Axis::Combined, Layer>,
    ChildNetwork: NeuraTrainableNetwork<NeuraResidualInput<Data>, Optimizer>,
> NeuraTrainableNetwork<NeuraResidualInput<Data>, Optimizer> for NeuraResidualNode<Layer, ChildNetwork, Axis>
where
    Self: NeuraTrainableLayerBase<NeuraResidualInput<Data>, Gradient = (Layer::Gradient, Box<ChildNetwork::Gradient>)>,
{
    fn traverse(
        &self,
        input: &NeuraResidualInput<Data>,
        optimizer: &Optimizer,
    ) -> Optimizer::Output<NeuraResidualInput<Data>, Self::Gradient> {
        let (layer_input, mut rest) = self.process_input(input);
        let (layer_output, layer_intermediary) = self.layer.eval_training(&layer_input);
        let layer_output = self.combine_outputs(layer_output, &mut rest);

        let child_result = self.child_network.traverse(&rest, optimizer);
        // TODO: maybe move this to a custom impl of NeuraGradientSolverTransient for NeuraResidualInput?
        // Or have a different set of traits for NeuraTrainableNetwork specific to NeuraResidualNodes
        let child_result = optimizer.map_epsilon(child_result, |epsilon| {
            // Pop the first value from `epsilon`, then:
            // - compute its sum
            // - use it to compute the outcoming epsilon of the current layer
            // - split the oucoming epsilon into its original components, and push those back onto the rest
            // At this point, the value for `epsilon` in the gradient solver's state should be ready for another iteration,
            // with the first value containing the unsummed incoming epsilon values from the downstream layers
            todo!();
        });

        todo!();
    }
}
