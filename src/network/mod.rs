use crate::{algebra::NeuraVectorSpace, derivable::NeuraLoss, layer::NeuraLayer};

pub mod sequential;

pub trait NeuraTrainableNetwork<Input>: NeuraLayer<Input> {
    type Gradient: NeuraVectorSpace;

    fn default_gradient(&self) -> Self::Gradient;

    fn apply_gradient(&mut self, gradient: &Self::Gradient);

    /// Should implement the backpropagation algorithm, see `NeuraTrainableLayer::backpropagate` for more information.
    fn backpropagate<Loss: NeuraLoss<Input = Self::Output>>(
        &self,
        input: &Input,
        target: &Loss::Target,
        loss: Loss,
    ) -> (Input, Self::Gradient);

    /// Should return the regularization gradient
    fn regularize(&self) -> Self::Gradient;

    /// Called before an iteration begins, to allow the network to set itself up for training or not.
    fn prepare(&mut self, train_iteration: bool);
}
