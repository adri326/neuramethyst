use crate::{
    algebra::NeuraVectorSpace,
    gradient_solver::{NeuraGradientSolverBase, NeuraGradientSolverFinal},
    layer::NeuraLayer,
};

// pub mod residual;
pub mod sequential;

mod traits;
pub use traits::*;

// TODO: extract regularize from this, so that we can drop the trait constraints on NeuraSequential's impl
pub trait NeuraOldTrainableNetworkBase<Input>: NeuraLayer<Input> {
    type Gradient: NeuraVectorSpace;
    type LayerOutput;

    fn default_gradient(&self) -> Self::Gradient;

    fn apply_gradient(&mut self, gradient: &Self::Gradient);

    /// Should return the regularization gradient
    fn regularize(&self) -> Self::Gradient;

    /// Called before an iteration begins, to allow the network to set itself up for training or not.
    fn prepare(&mut self, train_iteration: bool);
}

pub trait NeuraOldTrainableNetwork<Input, Optimizer>: NeuraOldTrainableNetworkBase<Input>
where
    Optimizer: NeuraGradientSolverBase,
{
    fn traverse(
        &self,
        input: &Input,
        optimizer: &Optimizer,
    ) -> Optimizer::Output<Input, Self::Gradient>;
}

impl<Input: Clone, Optimizer: NeuraGradientSolverFinal<Input>>
    NeuraOldTrainableNetwork<Input, Optimizer> for ()
{
    fn traverse(
        &self,
        input: &Input,
        optimizer: &Optimizer,
    ) -> Optimizer::Output<Input, Self::Gradient> {
        optimizer.eval_final(input.clone())
    }
}
