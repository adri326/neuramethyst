use nalgebra::DVector;

use crate::algebra::NeuraVector;

use super::NeuraLoss;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Euclidean;

impl NeuraLoss for Euclidean {
    type Input = DVector<f64>;
    type Target = DVector<f64>;

    #[inline]
    fn eval(&self, target: &DVector<f64>, actual: &DVector<f64>) -> f64 {
        assert_eq!(target.shape(), actual.shape());
        let mut sum_squared = 0.0;

        for i in 0..target.len() {
            sum_squared += (target[i] - actual[i]) * (target[i] - actual[i]);
        }

        sum_squared * 0.5
    }

    #[inline]
    fn nabla(
        &self,
        target: &DVector<f64>,
        actual: &DVector<f64>,
    ) -> DVector<f64> {
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
pub struct CrossEntropy<const N: usize>;

const DERIVATIVE_CAP: f64 = 100.0;
const LOG_MIN: f64 = 0.00001;

impl<const N: usize> CrossEntropy<N> {
    #[inline(always)]
    pub fn eval_single(&self, target: f64, actual: f64) -> f64 {
        -target * actual.max(LOG_MIN).log(std::f64::consts::E)
    }

    #[inline(always)]
    pub fn derivate_single(&self, target: f64, actual: f64) -> f64 {
        -(target / actual).min(DERIVATIVE_CAP)
    }
}

impl<const N: usize> NeuraLoss for CrossEntropy<N> {
    type Input = NeuraVector<N, f64>;
    type Target = NeuraVector<N, f64>;

    fn eval(&self, target: &Self::Target, actual: &Self::Input) -> f64 {
        let mut result = 0.0;

        for i in 0..N {
            result += self.eval_single(target[i], actual[i]);
        }

        result
    }

    fn nabla(&self, target: &Self::Target, actual: &Self::Input) -> Self::Input {
        let mut result = NeuraVector::default();

        for i in 0..N {
            result[i] = self.derivate_single(target[i], actual[i]);
        }

        result
    }
}

// TODO: a one-hot encoded, CrossEntropy + Softmax loss function?
// It would be a lot more efficient than the current method
