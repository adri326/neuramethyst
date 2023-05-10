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

impl<R: Rng + Clone + std::fmt::Debug + 'static> NeuraPartialLayer for NeuraDropoutLayer<R> {
    type Constructed = NeuraDropoutLayer<R>;

    type Err = ();

    fn construct(mut self, input_shape: NeuraShape) -> Result<Self::Constructed, Self::Err> {
        self.shape = input_shape;
        self.mask = DVector::from_element(input_shape.size(), false);
        Ok(self)
    }
}

impl<R: Rng + Clone + std::fmt::Debug + 'static> NeuraLayerBase for NeuraDropoutLayer<R> {
    type Gradient = ();

    fn default_gradient(&self) -> Self::Gradient {
        
    }

    fn output_shape(&self) -> NeuraShape {
        self.shape
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

impl<R: Rng + Clone + std::fmt::Debug + 'static, F: Float> NeuraLayer<DVector<F>>
    for NeuraDropoutLayer<R>
{
    type Output = DVector<F>;

    type IntermediaryRepr = ();

    fn eval_training(&self, input: &DVector<F>) -> (Self::Output, Self::IntermediaryRepr) {
        let mut output = input.clone();
        self.apply_dropout(&mut output);
        (output, ())
    }

    fn backprop_layer(
        &self,
        _input: &DVector<F>,
        _intermediary: &Self::IntermediaryRepr,
        epsilon: &Self::Output,
    ) -> DVector<F> {
        let mut epsilon = epsilon.clone();

        self.apply_dropout(&mut epsilon);

        epsilon
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
            layer.prepare_layer(true);
            assert!(layer.multiplier.is_finite());
            assert!(!layer.multiplier.is_nan());
        }
    }
}
