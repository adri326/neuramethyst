use super::NeuraLoss;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Euclidean<const N: usize>;

impl<const N: usize> NeuraLoss for Euclidean<N> {
    type Input = [f64; N];
    type Target = [f64; N];

    #[inline]
    fn eval(&self, target: &[f64; N], actual: &[f64; N]) -> f64 {
        let mut sum_squared = 0.0;

        for i in 0..N {
            sum_squared += (target[i] - actual[i]) * (target[i] - actual[i]);
        }

        sum_squared * 0.5
    }

    #[inline]
    fn nabla(&self, target: &[f64; N], actual: &[f64; N]) -> [f64; N] {
        let mut res = [0.0; N];

        // ∂E(y)/∂yᵢ = yᵢ - yᵢ'
        for i in 0..N {
            res[i] = actual[i] - target[i];
        }

        res
    }
}
