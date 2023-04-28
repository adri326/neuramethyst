mod backprop;
pub use backprop::NeuraBackprop;

mod forward_forward;
pub use forward_forward::NeuraForwardForward;

use crate::layer::{NeuraTrainableLayerBase, NeuraTrainableLayerEval};

pub trait NeuraGradientSolver<Input, Target, Trainable: NeuraTrainableLayerBase> {
    fn get_gradient(
        &self,
        trainable: &Trainable,
        input: &Input,
        target: &Target,
    ) -> Trainable::Gradient;

    fn score(&self, trainable: &Trainable, input: &Input, target: &Target) -> f64;
}
