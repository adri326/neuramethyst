//! This module is currently disabled, as it relies on `generic_const_exprs`, which is too unstable to use as of now

use super::{NeuraLayer, NeuraTrainableLayer};

/// Converts a `[[T; WIDTH]; HEIGHT]` into a `[T; WIDTH * HEIGHT]`.
/// Requires the `#![feature(generic_const_exprs)]` feature to be enabled.
pub struct NeuraFlattenLayer<const WIDTH: usize, const HEIGHT: usize, T> {
    phantom: std::marker::PhantomData<T>,
}

/// Converts a `[T; WIDTH * HEIGHT]` into a `[[T; WIDTH]; HEIGHT]`.
/// Requires the `#![feature(generic_const_exprs)]` feature to be enabled.
pub struct NeuraReshapeLayer<const WIDTH: usize, const HEIGHT: usize, T> {
    phantom: std::marker::PhantomData<T>,
}

#[inline(always)]
fn flatten<const WIDTH: usize, const HEIGHT: usize, T: Copy + Default>(
    input: &[[T; WIDTH]; HEIGHT],
) -> [T; WIDTH * HEIGHT]
where
    [T; WIDTH * HEIGHT]: Sized,
{
    let mut res = [T::default(); WIDTH * HEIGHT];

    // Hopefully the optimizer realizes this can be all optimized away
    for i in 0..HEIGHT {
        for j in 0..WIDTH {
            res[i * WIDTH + j] = input[i][j];
        }
    }

    res
}

#[inline(always)]
fn reshape<const WIDTH: usize, const HEIGHT: usize, T: Copy + Default>(
    input: &[T; WIDTH * HEIGHT],
) -> [[T; WIDTH]; HEIGHT]
where
    [T; WIDTH * HEIGHT]: Sized,
{
    let mut res = [[T::default(); WIDTH]; HEIGHT];

    // Hopefully the optimizer realizes this can be all optimized away
    for i in 0..HEIGHT {
        for j in 0..WIDTH {
            res[i][j] = input[i * WIDTH + j];
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
    [T; WIDTH * HEIGHT]: Sized,
{
    type Input = [[T; WIDTH]; HEIGHT];

    type Output = [T; WIDTH * HEIGHT];

    #[inline(always)]
    fn eval(&self, input: &Self::Input) -> Self::Output {
        flatten(input)
    }
}

impl<const WIDTH: usize, const HEIGHT: usize, T: Copy + Default> NeuraLayer
    for NeuraReshapeLayer<WIDTH, HEIGHT, T>
where
    [T; WIDTH * HEIGHT]: Sized,
{
    type Input = [T; WIDTH * HEIGHT];

    type Output = [[T; WIDTH]; HEIGHT];

    #[inline(always)]
    fn eval(&self, input: &Self::Input) -> Self::Output {
        reshape(input)
    }
}

impl<const WIDTH: usize, const HEIGHT: usize, T: Copy + Default> NeuraTrainableLayer
    for NeuraFlattenLayer<WIDTH, HEIGHT, T>
where
    [T; WIDTH * HEIGHT]: Sized,
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
        todo!()
    }

    fn apply_gradient(&mut self, _gradient: &Self::Delta) {
        // Noop
    }
}

impl<const WIDTH: usize, const HEIGHT: usize, T: Copy + Default> NeuraTrainableLayer
    for NeuraReshapeLayer<WIDTH, HEIGHT, T>
where
    [T; WIDTH * HEIGHT]: Sized,
{
    type Delta = ();

    fn backpropagate(
        &self,
        _input: &Self::Input,
        epsilon: Self::Output,
    ) -> (Self::Input, Self::Delta) {
        (flatten(&epsilon), ())
    }

    fn regularize(&self) -> Self::Delta {
        todo!()
    }

    fn apply_gradient(&mut self, _gradient: &Self::Delta) {
        // Noop
    }
}
