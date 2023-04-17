use crate::utils::{argmax, one_hot};

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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Max;

impl NeuraReducer<f64> for Max {
    #[inline(always)]
    fn eval<const LENGTH: usize>(&self, inputs: NeuraVector<LENGTH, f64>) -> f64 {
        let mut max = 0.0;
        for &i in inputs.iter() {
            max = i.max(max);
        }
        max
    }

    #[inline(always)]
    fn nabla<const LENGTH: usize>(
        &self,
        inputs: NeuraVector<LENGTH, f64>,
    ) -> NeuraVector<LENGTH, f64> {
        one_hot(argmax(inputs.as_ref()))
    }
}
