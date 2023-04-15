use super::{NeuraLayer, NeuraTrainableLayer};

/// A special layer that allows you to split a vector into one-hot vectors
#[derive(Debug, Clone, PartialEq)]
pub struct NeuraOneHotLayer<const CATS: usize, const LENGTH: usize>;

impl<const CATS: usize, const LENGTH: usize> NeuraLayer for NeuraOneHotLayer<CATS, LENGTH>
where
    [(); LENGTH * CATS]: Sized,
{
    type Input = [f64; LENGTH];
    type Output = [f64; LENGTH * CATS];

    fn eval(&self, input: &Self::Input) -> Self::Output {
        let mut res = [0.0; LENGTH * CATS];

        for i in 0..LENGTH {
            let cat_low = input[i].floor().max(0.0).min(CATS as f64 - 2.0);
            let amount = (input[i] - cat_low).max(0.0).min(1.0);
            let cat_low = cat_low as usize;
            res[i * LENGTH + cat_low] = 1.0 - amount;
            res[i * LENGTH + cat_low + 1] = amount;
        }

        res
    }
}

impl<const CATS: usize, const LENGTH: usize> NeuraTrainableLayer for NeuraOneHotLayer<CATS, LENGTH>
where
    [(); LENGTH * CATS]: Sized,
{
    type Delta = ();

    fn backpropagate(
        &self,
        input: &Self::Input,
        epsilon: Self::Output,
    ) -> (Self::Input, Self::Delta) {
        let mut res = [0.0; LENGTH];

        for i in 0..LENGTH {
            let cat_low = input[i].floor().max(0.0).min(CATS as f64 - 2.0) as usize;
            let epsilon = -epsilon[i * LENGTH + cat_low] + epsilon[i * LENGTH + cat_low + 1];
            // Scale epsilon by how many entries were ignored
            res[i] = epsilon * CATS as f64 / 2.0;
        }

        (res, ())
    }

    fn regularize(&self) -> Self::Delta {
        ()
    }

    fn apply_gradient(&mut self, _gradient: &Self::Delta) {
        // Noop
    }
}
