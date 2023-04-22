mod backprop;
pub use backprop::NeuraBackprop;

mod forward_forward;
pub use forward_forward::NeuraForwardForward;

use crate::{
    layer::NeuraTrainableLayerBase,
    network::{NeuraTrainableNetwork, NeuraTrainableNetworkBase},
};

pub trait NeuraGradientSolverBase {
    type Output<NetworkInput, NetworkGradient>;
}

pub trait NeuraGradientSolverFinal<LayerOutput>: NeuraGradientSolverBase {
    fn eval_final(&self, output: LayerOutput) -> Self::Output<LayerOutput, ()>;
}

pub trait NeuraGradientSolverTransient<Input, Layer: NeuraTrainableLayerBase<Input>>:
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
}

pub trait NeuraGradientSolver<Input, Target, Trainable: NeuraTrainableNetworkBase<Input>> {
    fn get_gradient(
        &self,
        trainable: &Trainable,
        input: &Input,
        target: &Target,
    ) -> Trainable::Gradient;

    fn score(&self, trainable: &Trainable, input: &Input, target: &Target) -> f64;
}
