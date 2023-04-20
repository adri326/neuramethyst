use crate::{algebra::NeuraVectorSpace, layer::NeuraLayer, optimize::NeuraOptimizerBase};

pub mod sequential;

pub trait NeuraTrainableNetworkBase<Input>: NeuraLayer<Input> {
    type Gradient: NeuraVectorSpace;
    type LayerOutput;

    fn default_gradient(&self) -> Self::Gradient;

    fn apply_gradient(&mut self, gradient: &Self::Gradient);

    /// Should return the regularization gradient
    fn regularize(&self) -> Self::Gradient;

    /// Called before an iteration begins, to allow the network to set itself up for training or not.
    fn prepare(&mut self, train_iteration: bool);
}

pub trait NeuraTrainableNetwork<Input, Optimizer>: NeuraTrainableNetworkBase<Input>
where
    Optimizer: NeuraOptimizerBase,
{
    fn traverse(
        &self,
        input: &Input,
        optimizer: &Optimizer,
    ) -> Optimizer::Output<Input, Self::Gradient>;
}
