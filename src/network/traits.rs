use std::borrow::Cow;

use crate::prelude::NeuraTrainableLayerBase;

use super::*;

/// This trait has to be non-generic, to ensure that no downstream crate can implement it for foreign types,
/// as that would otherwise cause infinite recursion when dealing with `NeuraNetworkRec`.
pub trait NeuraNetworkBase {
    /// The type of the enclosed layer
    type Layer;

    fn get_layer(&self) -> &Self::Layer;
}

pub trait NeuraNetwork<NodeInput: Clone>: NeuraNetworkBase
where
    Self::Layer: NeuraLayer<Self::LayerInput>,
    <Self::Layer as NeuraLayer<Self::LayerInput>>::Output: Clone,
{
    /// The type of the input to `Self::Layer`
    type LayerInput: Clone;

    /// The type of the output of this node
    type NodeOutput: Clone;

    /// Maps the input of network node to the enclosed layer
    fn map_input<'a>(&'_ self, input: &'a NodeInput) -> Cow<'a, Self::LayerInput>;
    /// Maps the output of the enclosed layer to the output of the network node
    fn map_output<'a>(
        &'_ self,
        input: &'_ NodeInput,
        layer_output: &'a <Self::Layer as NeuraLayer<Self::LayerInput>>::Output,
    ) -> Cow<'a, Self::NodeOutput>;

    /// Maps a gradient in the format of the node's output into the format of the enclosed layer's output
    fn map_gradient_in<'a>(
        &'_ self,
        input: &'_ NodeInput,
        gradient_in: &'a Self::NodeOutput,
    ) -> Cow<'a, <Self::Layer as NeuraLayer<Self::LayerInput>>::Output>;
    /// Maps a gradient in the format of the enclosed layer's input into the format of the node's input
    fn map_gradient_out<'a>(
        &'_ self,
        input: &'_ NodeInput,
        gradient_in: &'_ Self::NodeOutput,
        gradient_out: &'a Self::LayerInput,
    ) -> Cow<'a, NodeInput>;
}

pub trait NeuraNetworkRec: NeuraNetworkBase + NeuraTrainableLayerBase {
    /// The type of the children network, it does not need to implement `NeuraNetworkBase`,
    /// although many functions will expect it to be either `()` or an implementation of `NeuraNetworkRec`.
    type NextNode: NeuraTrainableLayerBase;

    fn get_next(&self) -> &Self::NextNode;

    fn merge_gradient(
        &self,
        rec_gradient: <Self::NextNode as NeuraTrainableLayerBase>::Gradient,
        layer_gradient: <Self::Layer as NeuraTrainableLayerBase>::Gradient
    ) -> Self::Gradient
    where Self::Layer: NeuraTrainableLayerBase;
}
