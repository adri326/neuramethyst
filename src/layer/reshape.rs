//! This module requires the `generic_const_exprs` feature to be enabled,
//! which is still quite unstable as of writing this.

use std::borrow::Borrow;

use crate::algebra::{NeuraMatrix, NeuraVector};

use super::{NeuraLayer, NeuraTrainableLayer};

/// Converts a `[[T; WIDTH]; HEIGHT]` into a `NeuraVector<{WIDTH * HEIGHT}, T>`.
/// Requires the `#![feature(generic_const_exprs)]` feature to be enabled.
pub struct NeuraFlattenLayer<const WIDTH: usize, const HEIGHT: usize, T> {
    phantom: std::marker::PhantomData<T>,
}

/// Converts a `NeuraVector<{WIDTH * HEIGHT}, T>` into a `[[T; WIDTH]; HEIGHT]`.
/// Requires the `#![feature(generic_const_exprs)]` feature to be enabled.
pub struct NeuraReshapeLayer<const WIDTH: usize, const HEIGHT: usize, T> {
    phantom: std::marker::PhantomData<T>,
}

#[inline(always)]
fn flatten<const WIDTH: usize, const HEIGHT: usize, T: Clone + Default>(
    input: impl Borrow<[[T; WIDTH]; HEIGHT]>,
) -> NeuraVector<{ WIDTH * HEIGHT }, T> {
    let mut res = NeuraVector::default();
    let input = input.borrow();

    // Hopefully the optimizer realizes this can be all optimized away
    for i in 0..HEIGHT {
        for j in 0..WIDTH {
            res[i * WIDTH + j] = input[i][j].clone();
        }
    }

    res
}

#[inline(always)]
fn reshape<const WIDTH: usize, const HEIGHT: usize, T: Clone + Default>(
    input: impl Borrow<[T; WIDTH * HEIGHT]>,
) -> NeuraMatrix<WIDTH, HEIGHT, T> {
    let input = input.borrow();
    let mut res = NeuraMatrix::default();

    // Hopefully the optimizer realizes this can be all optimized away
    for i in 0..HEIGHT {
        for j in 0..WIDTH {
            res[i][j] = input[i * WIDTH + j].clone();
        }
    }

    res
}

impl<const WIDTH: usize, const HEIGHT: usize, T> NeuraFlattenLayer<WIDTH, HEIGHT, T> {
    pub fn new() -> Self {
        Self {
            phantom: std::marker::PhantomData,
        }
    }
}

impl<const WIDTH: usize, const HEIGHT: usize, T> NeuraReshapeLayer<WIDTH, HEIGHT, T> {
    pub fn new() -> Self {
        Self {
            phantom: std::marker::PhantomData,
        }
    }
}

impl<const WIDTH: usize, const HEIGHT: usize, T: Copy + Default> NeuraLayer
    for NeuraFlattenLayer<WIDTH, HEIGHT, T>
where
    NeuraVector<{ WIDTH * HEIGHT }, T>: Sized,
{
    type Input = NeuraMatrix<WIDTH, HEIGHT, T>;

    type Output = NeuraVector<{ WIDTH * HEIGHT }, T>;

    #[inline(always)]
    fn eval(&self, input: &Self::Input) -> Self::Output {
        flatten(input.as_ref())
    }
}

impl<const WIDTH: usize, const HEIGHT: usize, T: Copy + Default> NeuraLayer
    for NeuraReshapeLayer<WIDTH, HEIGHT, T>
where
    NeuraVector<{ WIDTH * HEIGHT }, T>: Sized,
{
    type Input = NeuraVector<{ WIDTH * HEIGHT }, T>;

    type Output = NeuraMatrix<WIDTH, HEIGHT, T>;

    #[inline(always)]
    fn eval(&self, input: &Self::Input) -> Self::Output {
        reshape(input)
    }
}

impl<const WIDTH: usize, const HEIGHT: usize, T: Copy + Default> NeuraTrainableLayer
    for NeuraFlattenLayer<WIDTH, HEIGHT, T>
where
    NeuraVector<{ WIDTH * HEIGHT }, T>: Sized,
{
    type Delta = ();

    fn backpropagate(
        &self,
        _input: &Self::Input,
        epsilon: Self::Output,
    ) -> (Self::Input, Self::Delta) {
        (reshape(&epsilon), ())
    }

    fn regularize(&self) -> Self::Delta {
        ()
    }

    fn apply_gradient(&mut self, _gradient: &Self::Delta) {
        // Noop
    }
}

impl<const WIDTH: usize, const HEIGHT: usize, T: Copy + Default> NeuraTrainableLayer
    for NeuraReshapeLayer<WIDTH, HEIGHT, T>
where
    NeuraVector<{ WIDTH * HEIGHT }, T>: Sized,
{
    type Delta = ();

    fn backpropagate(
        &self,
        _input: &Self::Input,
        epsilon: Self::Output,
    ) -> (Self::Input, Self::Delta) {
        (flatten(epsilon), ())
    }

    fn regularize(&self) -> Self::Delta {
        ()
    }

    fn apply_gradient(&mut self, _gradient: &Self::Delta) {
        // Noop
    }
}
