use crate::{algebra::NeuraVectorSpace, derivable::NeuraLoss, layer::NeuraLayer};

pub mod sequential;

pub trait NeuraTrainableNetwork<Input>: NeuraLayer<Input> {
    type Delta: NeuraVectorSpace;

    fn default_gradient(&self) -> Self::Delta;

    fn apply_gradient(&mut self, gradient: &Self::Delta);

    /// Should implement the backpropagation algorithm, see `NeuraTrainableLayer::backpropagate` for more information.
    fn backpropagate<Loss: NeuraLoss<Input = Self::Output>>(
        &self,
        input: &Input,
        target: &Loss::Target,
        loss: Loss,
    ) -> (Input, Self::Delta);

    /// Should return the regularization gradient
    fn regularize(&self) -> Self::Delta;

    /// Called before an iteration begins, to allow the network to set itself up for training or not.
    fn prepare(&mut self, train_iteration: bool);
}
