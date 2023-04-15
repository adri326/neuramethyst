use crate::{algebra::NeuraVectorSpace, derivable::NeuraLoss, layer::NeuraLayer};

pub mod sequential;

pub trait NeuraTrainableNetwork: NeuraLayer {
    type Delta: NeuraVectorSpace;

    fn apply_gradient(&mut self, gradient: &Self::Delta);

    /// Should implement the backpropagation algorithm, see `NeuraTrainableLayer::backpropagate` for more information.
    fn backpropagate<Loss: NeuraLoss<Input = Self::Output>>(
        &self,
        input: &Self::Input,
        target: &Loss::Target,
        loss: Loss,
    ) -> (Self::Input, Self::Delta);

    /// Should return the regularization gradient
    fn regularize(&self) -> Self::Delta;

    /// Called before an iteration begins, to allow the network to set itself up for training.
    fn prepare_epoch(&mut self);

    /// Called at the end of training, to allow the network to clean itself up
    fn cleanup(&mut self);
}
