use super::NeuraDerivable;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Relu;

impl NeuraDerivable<f64> for Relu {
    #[inline(always)]
    fn eval(&self, input: f64) -> f64 {
        input.max(0.0)
    }

    #[inline(always)]
    fn derivate(&self, input: f64) -> f64 {
        if input > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

impl NeuraDerivable<f32> for Relu {
    #[inline(always)]
    fn eval(&self, input: f32) -> f32 {
        input.max(0.0)
    }

    #[inline(always)]
    fn derivate(&self, input: f32) -> f32 {
        if input > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}
