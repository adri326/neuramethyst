pub trait Activation {
    fn eval(&self, input: f64) -> f64;

    fn eval_f32(&self, input: f32) -> f32 {
        self.eval(input as f64) as f32
    }

    fn derivate(&self, at: f64) -> f64;

    fn derivate_f32(&self, at: f32) -> f32 {
        self.derivate(at as f64) as f32
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Relu;
impl Activation for Relu {
    #[inline(always)]
    fn eval(&self, input: f64) -> f64 {
        input.max(0.0)
    }

    #[inline(always)]
    fn eval_f32(&self, input: f32) -> f32 {
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

    #[inline(always)]
    fn derivate_f32(&self, input: f32) -> f32 {
        if input > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}
