mod backprop;
pub use backprop::NeuraBackprop;

mod forward_forward;
pub use forward_forward::NeuraForwardForward;

use crate::{
    layer::{NeuraTrainableLayerBase, NeuraTrainableLayerEval},
    network::{NeuraOldTrainableNetwork, NeuraOldTrainableNetworkBase},
};

pub trait NeuraGradientSolverBase {
    type Output<NetworkInput, NetworkGradient>;
}

pub trait NeuraGradientSolverFinal<LayerOutput>: NeuraGradientSolverBase {
    fn eval_final(&self, output: LayerOutput) -> Self::Output<LayerOutput, ()>;
}

pub trait NeuraGradientSolverTransient<Input, Layer: NeuraTrainableLayerEval<Input>>:
    NeuraGradientSolverBase
{
    fn eval_layer<NetworkGradient, RecGradient>(
        &self,
        layer: &Layer,
        input: &Input,
        output: &Layer::Output,
        layer_intermediary: &Layer::IntermediaryRepr,
        rec_opt_output: Self::Output<Layer::Output, RecGradient>,
        combine_gradients: impl Fn(Layer::Gradient, RecGradient) -> NetworkGradient,
    ) -> Self::Output<Input, NetworkGradient>;

    fn map_epsilon<From, To, Gradient, Cb: Fn(From) -> To>(
        &self,
        rec_opt_output: Self::Output<From, Gradient>,
        callback: Cb,
    ) -> Self::Output<To, Gradient>;
}

pub trait NeuraGradientSolver<Input, Target, Trainable: NeuraTrainableLayerBase> {
    fn get_gradient(
        &self,
        trainable: &Trainable,
        input: &Input,
        target: &Target,
    ) -> Trainable::Gradient;

    fn score(&self, trainable: &Trainable, input: &Input, target: &Target) -> f64;
}
