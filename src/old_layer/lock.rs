use super::{NeuraLayer, NeuraTrainableLayer};

/// Represents a layer that has been locked:
/// it won't be modified during training and its weight won't be stored
#[derive(Clone, Debug, PartialEq)]
pub struct NeuraLockLayer<L: NeuraLayer>(pub L);

impl<L: NeuraLayer> NeuraLayer for NeuraLockLayer<L> {
    type Input = L::Input;

    type Output = L::Output;

    fn eval(&self, input: &Self::Input) -> Self::Output {
        self.0.eval(input)
    }
}

impl<L: NeuraLayer + NeuraTrainableLayer> NeuraTrainableLayer for NeuraLockLayer<L> {
    type Delta = ();

    #[inline(always)]
    fn backpropagate(
        &self,
        input: &Self::Input,
        epsilon: Self::Output,
    ) -> (Self::Input, Self::Delta) {
        (self.0.backpropagate(input, epsilon).0, ())
    }

    #[inline(always)]
    fn regularize(&self) -> Self::Delta {
        ()
    }

    #[inline(always)]
    fn apply_gradient(&mut self, _gradient: &Self::Delta) {
        // Noop
    }
}
