use super::*;
use nalgebra::DVector;
use num::Float;
use rand::Rng;

#[derive(Clone, Debug)]
pub struct NeuraDropoutLayer<R: Rng> {
    pub dropout_probability: f64,
    multiplier: f64,
    mask: DVector<bool>,
    rng: R,
    shape: NeuraShape,
}

impl<R: Rng> NeuraDropoutLayer<R> {
    pub fn new(dropout_probability: f64, rng: R) -> Self {
        Self {
            dropout_probability,
            multiplier: 1.0,
            mask: DVector::from_element(0, false),
            rng,
            shape: NeuraShape::Vector(0),
        }
    }

    fn apply_dropout<F: Float>(&self, vector: &mut DVector<F>) {
        let multiplier = F::from(self.multiplier).unwrap();
        for (index, &dropout) in self.mask.iter().enumerate() {
            if dropout {
                vector[index] = F::zero();
            } else {
                vector[index] = vector[index] * multiplier;
            }
        }
    }
}

impl<R: Rng> NeuraPartialLayer for NeuraDropoutLayer<R> {
    type Constructed = NeuraDropoutLayer<R>;

    type Err = ();

    fn construct(mut self, input_shape: NeuraShape) -> Result<Self::Constructed, Self::Err> {
        self.shape = input_shape;
        self.mask = DVector::from_element(input_shape.size(), false);
        Ok(self)
    }

    fn output_shape(constructed: &Self::Constructed) -> NeuraShape {
        constructed.shape
    }
}

impl<R: Rng, F: Float> NeuraLayer<DVector<F>> for NeuraDropoutLayer<R> {
    type Output = DVector<F>;

    fn eval(&self, input: &DVector<F>) -> Self::Output {
        let mut output = input.clone();
        self.apply_dropout(&mut output);
        output
    }
}

impl<R: Rng, F: Float> NeuraTrainableLayer<DVector<F>> for NeuraDropoutLayer<R> {
    type Gradient = ();

    fn default_gradient(&self) -> Self::Gradient {
        ()
    }

    fn backprop_layer(
        &self,
        _input: &DVector<F>,
        mut epsilon: Self::Output,
    ) -> (DVector<F>, Self::Gradient) {
        self.apply_dropout(&mut epsilon);

        (epsilon, ())
    }

    fn regularize_layer(&self) -> Self::Gradient {
        ()
    }

    fn apply_gradient(&mut self, _gradient: &Self::Gradient) {
        // Noop
    }

    fn prepare_layer(&mut self, is_training: bool) {
        let length = self.shape.size();
        if !is_training {
            self.mask = DVector::from_element(length, false);
            self.multiplier = 1.0;
            return;
        }

        // Rejection sampling to prevent all the inputs from being dropped out
        loop {
            let mut sum = 0;
            for i in 0..length {
                self.mask[i] = self.rng.gen_bool(self.dropout_probability);
                sum += self.mask[i] as usize;
            }

            if sum < length {
                self.multiplier = length as f64 / (length - sum) as f64;
                break;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_rejection_sampling() {
        let mut layer = NeuraDropoutLayer::new(0.9, rand::thread_rng())
            .construct(NeuraShape::Vector(1))
            .unwrap();

        for _ in 0..100 {
            <NeuraDropoutLayer<_> as NeuraTrainableLayer<DVector<f64>>>::prepare_layer(
                &mut layer, true,
            );
            assert!(layer.multiplier.is_finite());
            assert!(!layer.multiplier.is_nan());
        }
    }
}
