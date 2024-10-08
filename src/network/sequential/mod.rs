use std::borrow::Cow;

use super::*;
use crate::layer::{NeuraLayer, NeuraLayerBase, NeuraPartialLayer, NeuraShape};

mod construct;
mod layer_impl;
mod lock;
mod tail;

pub use construct::*;
pub use lock::*;
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
/// `NeuraLayer` and `NeuraLayerBase` each require that `ChildNetwork` implements those respective traits,
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
/// until an instance of `NeuraSequential<Layer, NeuraSequentialLast>` is found.
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
            layer,
            child_network: Box::new(self),
        }
    }
}

impl<Layer> From<Layer> for NeuraSequential<Layer, NeuraSequentialLast> {
    fn from(layer: Layer) -> Self {
        Self {
            layer,
            child_network: Box::<NeuraSequentialLast>::default(),
        }
    }
}

impl<Layer, ChildNetwork> NeuraNetworkBase for NeuraSequential<Layer, ChildNetwork> {
    type Layer = Layer;

    fn get_layer(&self) -> &Self::Layer {
        &self.layer
    }
}

impl<Layer: NeuraLayerBase, ChildNetwork: NeuraLayerBase> NeuraNetworkRec
    for NeuraSequential<Layer, ChildNetwork>
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

impl<Input: Clone, Layer: NeuraLayer<Input>, ChildNetwork> NeuraNetwork<Input>
    for NeuraSequential<Layer, ChildNetwork>
where
    Layer::Output: Clone,
{
    type LayerInput = Input;
    type NodeOutput = Layer::Output;

    fn map_input<'a>(&'_ self, input: &'a Input) -> Cow<'a, Input> {
        Cow::Borrowed(input)
    }

    fn map_output<'a>(
        &'_ self,
        _input: &'_ Input,
        layer_output: &'a <Self::Layer as NeuraLayer<Input>>::Output,
    ) -> Cow<'a, Self::NodeOutput> {
        Cow::Borrowed(layer_output)
    }

    fn map_gradient_in<'a>(
        &'_ self,
        _input: &'_ Input,
        gradient_in: &'a Self::NodeOutput,
    ) -> Cow<'a, <Self::Layer as NeuraLayer<Input>>::Output> {
        Cow::Borrowed(gradient_in)
    }

    fn map_gradient_out<'a>(
        &'_ self,
        _input: &'_ Input,
        _gradient_in: &'_ Self::NodeOutput,
        gradient_out: &'a Self::LayerInput,
    ) -> Cow<'a, Input> {
        Cow::Borrowed(gradient_out)
    }
}

/// An utility to recursively create a NeuraSequential network, while writing it in a declarative and linear fashion.
/// Note that this can quickly create big and unwieldly types.
#[macro_export]
macro_rules! neura_sequential {
    [] => {
        $crate::network::sequential::NeuraSequentialLast::default()
    };

    [ .. $network:expr $(,)? ] => {
        $network
    };

    [ .. $network:expr, $layer:expr $(, $($rest:expr),+ )? $(,)? ] => {
        neura_sequential![ .. (($network).push_tail($layer)), $( $( $rest ),+ )? ]
    };

    // [ $( $lhs:expr, )* $layer:expr, .. $network:expr $(, $($rhs:expr),* )?] => {
    //     neura_sequential![ $($lhs,)* .. (($network).push_front($layer)) $(, $($rhs),* )? ]
    // };

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
        prelude::NeuraPartialLayer,
    };

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
