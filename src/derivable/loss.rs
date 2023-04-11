use super::NeuraLoss;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Euclidean;
impl<const N: usize> NeuraLoss<[f64; N]> for Euclidean {
    type Out = f64;
    type Target = [f64; N];

    fn eval(&self, target: [f64; N], actual: [f64; N]) -> f64 {
        let mut sum_squared = 0.0;

        for i in 0..N {
            sum_squared += (target[i] - actual[i]) * (target[i] - actual[i]);
        }

        sum_squared * 0.5
    }

    fn nabla(&self, target: [f64; N], actual: [f64; N]) -> [f64; N] {
        todo!()
    }
}
