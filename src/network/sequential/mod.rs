use super::{NeuraTrainableNetwork, NeuraTrainableNetworkBase};
use crate::{
    layer::{NeuraLayer, NeuraPartialLayer, NeuraShape, NeuraTrainableLayer},
    optimize::{NeuraOptimizerFinal, NeuraOptimizerTransient},
};

mod construct;
mod tail;

pub use construct::*;
pub use tail::*;

/// Chains a layer with the rest of a neural network, in a fashion similar to a cartesian product,
/// while preserving all type information.
/// The type `Layer` represents the current layer of the neural network,
/// and its output will be fed to the `ChildNetwork`, which will typically either be another `NeuraSequential`
/// instance or `()`.
///
/// `ChildNetwork` is also free to be another implementation of `NeuraNetwork`,
/// which allows `NeuraSequential` to be used together with other network structures.
///
/// `child_network` is stored in a `Box`, so as to avoid taking up too much space on the stack.
///
/// ## Notes on implemented traits
///
/// The different implementations for `NeuraTrainableNetwork`,
/// `NeuraLayer` and `NeuraTrainableLayer` each require that `ChildNetwork` implements those respective traits,
/// and that the output type of `Layer` matches the input type of `ChildNetwork`.
///
/// If a method, like `eval`, is reported as missing,
/// then it likely means that the output type of `Layer` does not match the input type of `ChildNetwork`,
/// or that a similar issue arose within `ChildNetwork`.
///
/// ## Trimming and appending layers
///
/// If you want to modify the network structure, you can do so by using the `trim_front`, `trim_tail`,
/// `push_front` and `push_tail` methods.
///
/// The operations on the front are trivial, as it simply involves wrapping the current instance in a new `NeuraSequential`
/// instance.
///
/// The operations on the tail end are more complex, and require recursively traversing the `NeuraSequential` structure,
/// until an instance of `NeuraSequential<Layer, ()>` is found.
/// If your network feeds into a type that does not implement `NeuraSequentialTail`, then you will not be able to use those operations.
#[derive(Clone, Debug)]
pub struct NeuraSequential<Layer, ChildNetwork> {
    pub layer: Layer,
    pub child_network: Box<ChildNetwork>,
}

impl<Layer, ChildNetwork> NeuraSequential<Layer, ChildNetwork> {
    pub fn new(layer: Layer, child_network: ChildNetwork) -> Self {
        Self {
            layer,
            child_network: Box::new(child_network),
        }
    }

    pub fn trim_front(self) -> ChildNetwork {
        *self.child_network
    }

    pub fn push_front<Input, Input2, T: NeuraLayer<Input2, Output = Input>>(
        self,
        layer: T,
    ) -> NeuraSequential<T, Self>
    where
        Layer: NeuraLayer<Input>,
    {
        NeuraSequential {
            layer: layer,
            child_network: Box::new(self),
        }
    }
}

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
        Layer: NeuraTrainableLayer<Input>,
        ChildNetwork: NeuraTrainableLayer<Layer::Output>,
    > NeuraTrainableLayer<Input> for NeuraSequential<Layer, ChildNetwork>
{
    type Gradient = (Layer::Gradient, Box<ChildNetwork::Gradient>);

    fn default_gradient(&self) -> Self::Gradient {
        (
            self.layer.default_gradient(),
            Box::new(self.child_network.default_gradient()),
        )
    }

    fn backprop_layer(
        &self,
        input: &Input,
        incoming_epsilon: Self::Output,
    ) -> (Input, Self::Gradient) {
        let output = self.layer.eval(input);
        let (transient_epsilon, child_gradient) =
            self.child_network.backprop_layer(&output, incoming_epsilon);
        let (outgoing_epsilon, layer_gradient) =
            self.layer.backprop_layer(input, transient_epsilon);

        (outgoing_epsilon, (layer_gradient, Box::new(child_gradient)))
    }

    fn regularize_layer(&self) -> Self::Gradient {
        (
            self.layer.regularize_layer(),
            Box::new(self.child_network.regularize_layer()),
        )
    }

    fn apply_gradient(&mut self, gradient: &Self::Gradient) {
        self.layer.apply_gradient(&gradient.0);
        self.child_network.apply_gradient(&gradient.1);
    }
}

impl<
        Input,
        Layer: NeuraTrainableLayer<Input>,
        ChildNetwork: NeuraTrainableNetworkBase<Layer::Output>,
    > NeuraTrainableNetworkBase<Input> for NeuraSequential<Layer, ChildNetwork>
{
    type Gradient = (Layer::Gradient, Box<ChildNetwork::Gradient>);
    type LayerOutput = Layer::Output;

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

    fn regularize(&self) -> Self::Gradient {
        (
            self.layer.regularize_layer(),
            Box::new(self.child_network.regularize()),
        )
    }

    fn prepare(&mut self, is_training: bool) {
        self.layer.prepare_layer(is_training);
        self.child_network.prepare(is_training);
    }
}

/// A dummy implementation of `NeuraTrainableNetwork`, which simply calls `loss.eval` in `backpropagate`.
impl<Input: Clone> NeuraTrainableNetworkBase<Input> for () {
    type Gradient = ();
    type LayerOutput = Input;

    #[inline(always)]
    fn default_gradient(&self) -> () {
        ()
    }

    #[inline(always)]
    fn apply_gradient(&mut self, _gradient: &()) {
        // Noop
    }

    #[inline(always)]
    fn regularize(&self) -> () {
        ()
    }

    #[inline(always)]
    fn prepare(&mut self, _is_training: bool) {
        // Noop
    }
}

impl<
        Input,
        Layer: NeuraTrainableLayer<Input>,
        Optimizer: NeuraOptimizerTransient<Layer::Output>,
        ChildNetwork: NeuraTrainableNetworkBase<Layer::Output>,
    > NeuraTrainableNetwork<Input, Optimizer> for NeuraSequential<Layer, ChildNetwork>
where
    ChildNetwork: NeuraTrainableNetwork<Layer::Output, Optimizer>,
{
    fn traverse(
        &self,
        input: &Input,
        optimizer: &Optimizer,
    ) -> Optimizer::Output<Input, Self::Gradient> {
        let next_activation = self.layer.eval(input);
        let child_result = self.child_network.traverse(&next_activation, optimizer);

        optimizer.eval_layer(
            &self.layer,
            input,
            child_result,
            |layer_gradient, child_gradient| (layer_gradient, Box::new(child_gradient)),
        )
    }
}

impl<Input: Clone, Optimizer: NeuraOptimizerFinal<Input>> NeuraTrainableNetwork<Input, Optimizer>
    for ()
{
    fn traverse(
        &self,
        input: &Input,
        optimizer: &Optimizer,
    ) -> Optimizer::Output<Input, Self::Gradient> {
        optimizer.eval_final(input.clone())
    }
}

impl<Layer> From<Layer> for NeuraSequential<Layer, ()> {
    fn from(layer: Layer) -> Self {
        Self {
            layer,
            child_network: Box::new(()),
        }
    }
}

/// An utility to recursively create a NeuraSequential network, while writing it in a declarative and linear fashion.
/// Note that this can quickly create big and unwieldly types.
#[macro_export]
macro_rules! neura_sequential {
    [] => {
        ()
    };

    [ $layer:expr $(,)? ] => {
        $crate::network::sequential::NeuraSequential::from($layer)
    };

    [ $first:expr, $($rest:expr),+ $(,)? ] => {
        $crate::network::sequential::NeuraSequential::new($first, neura_sequential![$($rest),+])
    };
}

#[cfg(test)]
mod test {
    use nalgebra::dvector;

    use crate::{
        derivable::{activation::Relu, regularize::NeuraL0},
        layer::{dense::NeuraDenseLayer, NeuraLayer, NeuraShape},
        neura_layer,
    };

    use super::NeuraSequentialConstruct;

    #[test]
    fn test_neura_network_macro() {
        let mut rng = rand::thread_rng();

        let _ = neura_sequential![
            NeuraDenseLayer::from_rng(8, 12, &mut rng, Relu, NeuraL0) as NeuraDenseLayer<f64, _, _>,
            NeuraDenseLayer::from_rng(12, 16, &mut rng, Relu, NeuraL0)
                as NeuraDenseLayer<f64, _, _>,
            NeuraDenseLayer::from_rng(16, 2, &mut rng, Relu, NeuraL0) as NeuraDenseLayer<f64, _, _>
        ];

        let _ =
            neura_sequential![NeuraDenseLayer::from_rng(2, 2, &mut rng, Relu, NeuraL0)
                as NeuraDenseLayer<f64, _, _>,];

        let _ = neura_sequential![
            NeuraDenseLayer::from_rng(8, 16, &mut rng, Relu, NeuraL0) as NeuraDenseLayer<f64, _, _>,
            NeuraDenseLayer::from_rng(16, 12, &mut rng, Relu, NeuraL0)
                as NeuraDenseLayer<f64, _, _>,
        ];

        let network = neura_sequential![
            neura_layer!("dense", 16).activation(Relu),
            neura_layer!("dense", 12).activation(Relu),
            neura_layer!("dense", 2).activation(Relu)
        ]
        .construct(NeuraShape::Vector(2))
        .unwrap();

        network.eval(&dvector![0.0, 0.0]);
    }
}
