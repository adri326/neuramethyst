mod backprop;
pub use backprop::NeuraBackprop;

mod forward_forward;
pub use forward_forward::NeuraForwardForward;

use crate::{
    layer::NeuraTrainableLayer,
    network::{NeuraTrainableNetwork, NeuraTrainableNetworkBase},
};

pub trait NeuraGradientSolverBase {
    type Output<NetworkInput, NetworkGradient>;
}

pub trait NeuraGradientSolverFinal<LayerOutput>: NeuraGradientSolverBase {
    fn eval_final(&self, output: LayerOutput) -> Self::Output<LayerOutput, ()>;
}

pub trait NeuraGradientSolverTransient<LayerOutput>: NeuraGradientSolverBase {
    fn eval_layer<
        Input,
        NetworkGradient,
        RecGradient,
        Layer: NeuraTrainableLayer<Input, Output = LayerOutput>,
    >(
        &self,
        layer: &Layer,
        input: &Input,
        rec_opt_output: Self::Output<LayerOutput, RecGradient>,
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
