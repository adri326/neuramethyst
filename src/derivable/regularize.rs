use super::*;

/// Default regularization, which is no regularization
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeuraL0;

impl NeuraDerivable<f64> for NeuraL0 {
    #[inline(always)]
    fn eval(&self, _input: f64) -> f64 {
        0.0
    }

    #[inline(always)]
    fn derivate(&self, _at: f64) -> f64 {
        0.0
    }
}

impl NeuraDerivable<f32> for NeuraL0 {
    #[inline(always)]
    fn eval(&self, _input: f32) -> f32 {
        0.0
    }

    #[inline(always)]
    fn derivate(&self, _at: f32) -> f32 {
        0.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeuraL1<F>(pub F);

impl NeuraDerivable<f64> for NeuraL1<f64> {
    #[inline(always)]
    fn eval(&self, input: f64) -> f64 {
        self.0 * input.abs()
    }

    #[inline(always)]
    fn derivate(&self, at: f64) -> f64 {
        if at > 0.0 {
            self.0
        } else if at < 0.0 {
            -self.0
        } else {
            0.0
        }
    }
}

impl NeuraDerivable<f32> for NeuraL1<f32> {
    #[inline(always)]
    fn eval(&self, input: f32) -> f32 {
        self.0 * input.abs()
    }

    #[inline(always)]
    fn derivate(&self, at: f32) -> f32 {
        if at > 0.0 {
            self.0
        } else if at < 0.0 {
            -self.0
        } else {
            0.0
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeuraL2<F>(pub F);

impl NeuraDerivable<f64> for NeuraL2<f64> {
    #[inline(always)]
    fn eval(&self, input: f64) -> f64 {
        self.0 * (input * input)
    }

    #[inline(always)]
    fn derivate(&self, at: f64) -> f64 {
        self.0 * at
    }
}

impl NeuraDerivable<f32> for NeuraL2<f32> {
    #[inline(always)]
    fn eval(&self, input: f32) -> f32 {
        self.0 * (input * input)
    }

    #[inline(always)]
    fn derivate(&self, at: f32) -> f32 {
        self.0 * at
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeuraElastic<F> {
    pub l1: F,
    pub l2: F,
}

impl<F> NeuraElastic<F> {
    pub fn new(l1_factor: F, l2_factor: F) -> Self {
        Self {
            l1: l1_factor,
            l2: l2_factor,
        }
    }
}

impl NeuraDerivable<f64> for NeuraElastic<f64> {
    #[inline(always)]
    fn eval(&self, input: f64) -> f64 {
        NeuraL1(self.l1).eval(input) + NeuraL2(self.l2).eval(input)
    }

    #[inline(always)]
    fn derivate(&self, at: f64) -> f64 {
        NeuraL1(self.l1).derivate(at) + NeuraL2(self.l2).derivate(at)
    }
}

impl NeuraDerivable<f32> for NeuraElastic<f32> {
    #[inline(always)]
    fn eval(&self, input: f32) -> f32 {
        NeuraL1(self.l1).eval(input) + NeuraL2(self.l2).eval(input)
    }

    #[inline(always)]
    fn derivate(&self, at: f32) -> f32 {
        NeuraL1(self.l1).derivate(at) + NeuraL2(self.l2).derivate(at)
    }
}
