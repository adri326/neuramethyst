use nalgebra::{DMatrix, DVector, Scalar};
use num::{traits::NumAssignOps, Float};

use super::*;

/// A layer that normalizes and centers its input, as follows:
///
/// ```no_rust
/// μ = sum_i(x_i) / n
/// σ = sum_i((x_i - μ)^2) / n
/// y_i = (x_i - μ) / σ
/// ```
#[derive(Debug, Clone, Copy)]
pub struct NeuraNormalizeLayer {
    shape: NeuraShape,
}

impl NeuraNormalizeLayer {
    pub fn new() -> Self {
        Self {
            shape: NeuraShape::Vector(0),
        }
    }
}

impl NeuraPartialLayer for NeuraNormalizeLayer {
    type Constructed = NeuraNormalizeLayer;

    type Err = ();

    fn construct(self, input_shape: NeuraShape) -> Result<Self::Constructed, Self::Err> {
        Ok(Self { shape: input_shape })
    }
}

impl NeuraLayerBase for NeuraNormalizeLayer {
    fn output_shape(&self) -> NeuraShape {
        self.shape
    }

    type Gradient = ();

    fn default_gradient(&self) -> Self::Gradient {
        
    }
}

impl<F: Float + Scalar + NumAssignOps> NeuraLayer<DVector<F>> for NeuraNormalizeLayer {
    // TODO: store the kroenecker term in the jacobian matrix (might as well)
    type IntermediaryRepr = (DMatrix<F>, F); // Partial jacobian matrix (without the kroenecker term) and stddev

    type Output = DVector<F>;

    fn eval(&self, input: &DVector<F>) -> Self::Output {
        let (mean, variance, _) = mean_variance(input);
        let stddev = F::sqrt(variance);

        let mut output = input.clone();

        for item in &mut output {
            *item = (*item - mean) / stddev;
        }

        output
    }

    fn eval_training(&self, input: &DVector<F>) -> (Self::Output, Self::IntermediaryRepr) {
        let (mean, variance, len) = mean_variance(input);
        let stddev = F::sqrt(variance);
        let input_centered = input.clone().map(|x| x - mean);

        let mut jacobian_partial = &input_centered * input_centered.transpose();
        jacobian_partial /= -variance * (stddev * len);
        // Apply the 1/σ * dμ/dx_i term
        for value in jacobian_partial.iter_mut() {
            *value += F::one() / (stddev * len);
        }

        (input_centered / stddev, (jacobian_partial, stddev))
    }

    fn backprop_layer(
        &self,
        _input: &DVector<F>,
        (jacobian_partial, stddev): &Self::IntermediaryRepr,
        epsilon: &Self::Output,
    ) -> DVector<F> {
        let mut epsilon_out = jacobian_partial * epsilon;

        // Apply the δ_{ik}/σ term
        for i in 0..epsilon_out.len() {
            epsilon_out[i] += epsilon[i] / *stddev;
        }

        epsilon_out
    }
}

fn mean_variance<'a, F: Float + Scalar>(input: impl IntoIterator<Item = &'a F>) -> (F, F, F) {
    // Quickly compute mean and variance in one pass
    let mut count = 0;
    let mut sum = F::zero();
    let mut sum_squared = F::zero();

    for &value in input.into_iter() {
        count += 1;
        sum = sum + value;
        sum_squared = sum_squared + value * value;
    }

    let len = F::from(count).unwrap();
    let mean = sum / len;
    let variance = sum_squared / len - mean * mean;

    (mean, variance, len)
}
