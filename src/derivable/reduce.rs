use super::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Average;

impl NeuraReducer<f64> for Average {
    #[inline(always)]
    fn eval<const LENGTH: usize>(&self, inputs: NeuraVector<LENGTH, f64>) -> f64 {
        let sum: f64 = inputs.iter().sum();
        sum / inputs.len() as f64
    }

    #[inline(always)]
    fn nabla<const LENGTH: usize>(
        &self,
        inputs: NeuraVector<LENGTH, f64>,
    ) -> NeuraVector<LENGTH, f64> {
        NeuraVector::from_value(1.0 / inputs.len() as f64)
    }
}
