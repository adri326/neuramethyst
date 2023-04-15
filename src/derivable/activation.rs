#![allow(unused_variables)]

use super::NeuraDerivable;

macro_rules! impl_derivable {
    ( $type_f32:ty, $type_f64:ty, $self:ident, $variable:ident, $eval:expr, $derivate:expr $(; $variance_hint:expr, $bias_hint:expr )? ) => {
        impl NeuraDerivable<f32> for $type_f32 {
            #[inline(always)]
            fn eval($self: &Self, $variable: f32) -> f32 {
                $eval
            }

            #[inline(always)]
            fn derivate($self: &Self, $variable: f32) -> f32 {
                $derivate
            }

            $(
                #[inline(always)]
                fn variance_hint($self: &Self) -> f64 {
                    $variance_hint
                }

                #[inline(always)]
                fn bias_hint($self: &Self) -> f64 {
                    $bias_hint
                }
            )?
        }

        impl NeuraDerivable<f64> for $type_f64 {
            #[inline(always)]
            fn eval($self: &Self, $variable: f64) -> f64 {
                $eval
            }

            #[inline(always)]
            fn derivate($self: &Self, $variable: f64) -> f64 {
                $derivate
            }

            $(
                #[inline(always)]
                fn variance_hint($self: &Self) -> f64 {
                    $variance_hint
                }

                #[inline(always)]
                fn bias_hint($self: &Self) -> f64 {
                    $bias_hint
                }
            )?
        }
    };

    ( $type:ty, $variable:ident, $eval:expr, $derivate:expr $(; $variance_hint:expr, $bias_hint:expr )? ) => {
        impl_derivable!($type, $type, self, $variable, $eval, $derivate $(; $variance_hint, $bias_hint)?);
    };
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Relu;

impl_derivable!(Relu, x, x.max(0.0), {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}; 2.0, 0.1);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LeakyRelu<F>(pub F);

impl_derivable!(
    LeakyRelu<f32>,
    LeakyRelu<f64>,
    self,
    x,
    {
        if x > 0.0 {
            x
        } else {
            self.0 * x
        }
    },
    {
        if x > 0.0 {
            1.0
        } else {
            self.0
        }
    };
    2.0, 0.1
);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tanh;

impl_derivable!(Tanh, x, x.tanh(), {
    let y = x.tanh();
    1.0 - y * y
});

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Linear;

impl_derivable!(Linear, x, x, 1.0);
