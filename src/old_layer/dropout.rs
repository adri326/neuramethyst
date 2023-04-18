use rand::Rng;

use crate::algebra::NeuraVector;

use super::{NeuraLayer, NeuraTrainableLayer};

#[derive(Clone, Debug)]
pub struct NeuraDropoutLayer<const LENGTH: usize, R: Rng> {
    pub dropout_probability: f64,
    multiplier: f64,
    mask: NeuraVector<LENGTH, bool>,
    rng: R,
}

impl<const LENGTH: usize, R: Rng> NeuraDropoutLayer<LENGTH, R> {
    pub fn new(dropout_probability: f64, rng: R) -> Self {
        Self {
            dropout_probability,
            multiplier: 1.0,
            mask: NeuraVector::from_value(false),
            rng,
        }
    }

    fn apply_dropout(&self, vector: &mut NeuraVector<LENGTH, f64>) {
        for (index, &dropout) in self.mask.iter().enumerate() {
            if dropout {
                vector[index] = 0.0;
            } else {
                vector[index] *= self.multiplier;
            }
        }
    }
}

impl<const LENGTH: usize, R: Rng> NeuraLayer for NeuraDropoutLayer<LENGTH, R> {
    type Input = NeuraVector<LENGTH, f64>;
    type Output = NeuraVector<LENGTH, f64>;

    fn eval(&self, input: &Self::Input) -> Self::Output {
        let mut result = input.clone();

        self.apply_dropout(&mut result);

        result
    }
}

impl<const LENGTH: usize, R: Rng> NeuraTrainableLayer for NeuraDropoutLayer<LENGTH, R> {
    type Delta = ();

    fn backpropagate(
        &self,
        _input: &Self::Input,
        mut epsilon: Self::Output,
    ) -> (Self::Input, Self::Delta) {
        self.apply_dropout(&mut epsilon);

        (epsilon, ())
    }

    fn regularize(&self) -> Self::Delta {
        ()
    }

    #[inline(always)]
    fn apply_gradient(&mut self, _gradient: &Self::Delta) {
        // Noop
    }

    fn prepare_epoch(&mut self) {
        // Rejection sampling to prevent all the inputs from being dropped out
        loop {
            let mut sum = 0;
            for i in 0..LENGTH {
                self.mask[i] = self.rng.gen_bool(self.dropout_probability);
                sum += (!self.mask[i]) as usize;
            }

            if sum < LENGTH {
                self.multiplier = LENGTH as f64 / sum as f64;
                break;
            }
        }
    }

    fn cleanup(&mut self) {
        self.mask = NeuraVector::from_value(false);
        self.multiplier = 1.0;
    }
}
