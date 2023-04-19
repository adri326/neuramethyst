use nalgebra::DVector;
use num::Float;

use crate::algebra::NeuraVector;

use super::NeuraLoss;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Euclidean;

impl<F: Float + std::fmt::Debug + 'static> NeuraLoss<DVector<F>> for Euclidean {
    type Target = DVector<F>;
    type Output = F;

    #[inline]
    fn eval(&self, target: &DVector<F>, actual: &DVector<F>) -> F {
        assert_eq!(target.shape(), actual.shape());
        let mut sum_squared = F::zero();

        for i in 0..target.len() {
            sum_squared = sum_squared + (target[i] - actual[i]) * (target[i] - actual[i]);
        }

        sum_squared * F::from(0.5).unwrap()
    }

    #[inline]
    fn nabla(&self, target: &DVector<F>, actual: &DVector<F>) -> DVector<F> {
        let mut res = DVector::zeros(target.len());

        // ∂E(y)/∂yᵢ = yᵢ - yᵢ'
        for i in 0..target.len() {
            res[i] = actual[i] - target[i];
        }

        res
    }
}

/// The cross-entropy loss function, defined as `L(y, ŷ) = -Σᵢ(yᵢ*ln(ŷᵢ))`.
///
/// This version of the cross-entropy function does not make assumptions about the target vector being one-hot encoded.
///
/// This function requires that `ŷ` (the output of the neural network) is in `[0; 1]^n`.
/// This guarantee is notably not given by the `Relu`, `LeakyRelu` and `Swish` activation functions,
/// so you should pick another activation on the last layer, or pass it into a `NeuraSoftmax` layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CrossEntropy;

const DERIVATIVE_CAP: f64 = 100.0;
const LOG_MIN: f64 = 0.00001;

impl CrossEntropy {
    #[inline(always)]
    pub fn eval_single<F: Float>(&self, target: F, actual: F) -> F {
        -target
            * actual
                .max(F::from(LOG_MIN).unwrap())
                .log(F::from(std::f64::consts::E).unwrap())
    }

    #[inline(always)]
    pub fn derivate_single<F: Float>(&self, target: F, actual: F) -> F {
        -(target / actual).min(F::from(DERIVATIVE_CAP).unwrap())
    }
}

impl<F: Float + std::fmt::Debug + 'static> NeuraLoss<DVector<F>> for CrossEntropy {
    type Target = DVector<F>;
    type Output = F;

    fn eval(&self, target: &Self::Target, actual: &DVector<F>) -> F {
        let mut result = F::zero();

        for i in 0..target.len() {
            result = result + self.eval_single(target[i], actual[i]);
        }

        result
    }

    fn nabla(&self, target: &Self::Target, actual: &DVector<F>) -> DVector<F> {
        let mut result = DVector::from_element(target.len(), F::zero());

        for i in 0..target.len() {
            result[i] = self.derivate_single(target[i], actual[i]);
        }

        result
    }
}

// TODO: a one-hot encoded, CrossEntropy + Softmax loss function?
// It would be a lot more efficient than the current method
