mod backprop;
pub use backprop::NeuraBackprop;

mod forward_forward;
pub use forward_forward::NeuraForwardForward;

use crate::layer::NeuraLayerBase;

pub trait NeuraGradientSolver<Input, Target, Trainable: NeuraLayerBase> {
    fn get_gradient(
        &self,
        trainable: &Trainable,
        input: &Input,
        target: &Target,
    ) -> Trainable::Gradient;

    fn score(&self, trainable: &Trainable, input: &Input, target: &Target) -> f64;
}
