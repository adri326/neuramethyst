use crate::{
    algebra::{NeuraMatrix, NeuraVector},
    derivable::NeuraReducer,
};

use super::*;

pub struct NeuraGlobalPoolLayer<const LENGTH: usize, const FEATS: usize, Reducer: NeuraReducer<f64>>
{
    reducer: Reducer,
}

impl<const LENGTH: usize, const FEATS: usize, Reducer: NeuraReducer<f64>>
    NeuraGlobalPoolLayer<LENGTH, FEATS, Reducer>
{
    pub fn new(reducer: Reducer) -> Self {
        Self { reducer }
    }
}

impl<const LENGTH: usize, const FEATS: usize, Reducer: NeuraReducer<f64>> NeuraLayer
    for NeuraGlobalPoolLayer<LENGTH, FEATS, Reducer>
{
    type Input = NeuraMatrix<FEATS, LENGTH, f64>;

    type Output = NeuraVector<FEATS, f64>;

    fn eval(&self, input: &Self::Input) -> Self::Output {
        let mut res = Self::Output::default();

        for j in 0..FEATS {
            let input = input.get_column(j);
            res[j] = self.reducer.eval(input);
        }

        res
    }
}

impl<const LENGTH: usize, const FEATS: usize, Reducer: NeuraReducer<f64>> NeuraTrainableLayer
    for NeuraGlobalPoolLayer<LENGTH, FEATS, Reducer>
{
    type Delta = ();

    #[inline]
    fn backpropagate(
        &self,
        input: &Self::Input,
        epsilon: Self::Output,
    ) -> (Self::Input, Self::Delta) {
        let mut next_epsilon = Self::Input::default();

        for j in 0..FEATS {
            let input = input.get_column(j);
            let mut gradient = self.reducer.nabla(input);
            gradient.mul_assign(epsilon[j]);
            next_epsilon.set_column(j, gradient);
        }

        (next_epsilon, ())
    }

    #[inline(always)]
    fn regularize(&self) -> Self::Delta {
        ()
    }

    #[inline(always)]
    fn apply_gradient(&mut self, _gradient: &Self::Delta) {
        // Noop
    }
}

pub struct NeuraPool1DLayer<
    const BLOCKS: usize,
    const BLOCK_LENGTH: usize,
    const FEATS: usize,
    Reducer: NeuraReducer<f64>,
> {
    reducer: Reducer,
}

impl<
        const BLOCKS: usize,
        const BLOCK_LENGTH: usize,
        const FEATS: usize,
        Reducer: NeuraReducer<f64>,
    > NeuraPool1DLayer<BLOCKS, BLOCK_LENGTH, FEATS, Reducer>
{
    pub fn new(reducer: Reducer) -> Self {
        Self { reducer }
    }
}

impl<
        const BLOCKS: usize,
        const BLOCK_LENGTH: usize,
        const FEATS: usize,
        Reducer: NeuraReducer<f64>,
    > NeuraLayer for NeuraPool1DLayer<BLOCKS, BLOCK_LENGTH, FEATS, Reducer>
where
    [f64; BLOCKS * BLOCK_LENGTH]: Sized,
{
    type Input = NeuraMatrix<FEATS, { BLOCKS * BLOCK_LENGTH }, f64>;
    type Output = NeuraMatrix<FEATS, BLOCKS, f64>;

    fn eval(&self, input: &Self::Input) -> Self::Output {
        let mut result = NeuraMatrix::default();

        for j in 0..FEATS {
            let input = input.get_column(j);
            for block in 0..BLOCKS {
                let mut block_input: NeuraVector<BLOCK_LENGTH, f64> = NeuraVector::default();
                for k in 0..BLOCK_LENGTH {
                    block_input[k] = input[block * BLOCK_LENGTH + k];
                }
                result[block][j] = self.reducer.eval(block_input);
            }
        }

        result
    }
}

impl<
        const BLOCKS: usize,
        const BLOCK_LENGTH: usize,
        const FEATS: usize,
        Reducer: NeuraReducer<f64>,
    > NeuraTrainableLayer for NeuraPool1DLayer<BLOCKS, BLOCK_LENGTH, FEATS, Reducer>
where
    [f64; BLOCKS * BLOCK_LENGTH]: Sized,
{
    type Delta = ();

    fn backpropagate(
        &self,
        input: &Self::Input,
        epsilon: Self::Output,
    ) -> (Self::Input, Self::Delta) {
        let mut next_epsilon = Self::Input::default();

        for j in 0..FEATS {
            let input = input.get_column(j);
            let mut column_gradient = NeuraVector::default();

            for block in 0..BLOCKS {
                let mut block_input: NeuraVector<BLOCK_LENGTH, f64> = NeuraVector::default();
                for k in 0..BLOCK_LENGTH {
                    block_input[k] = input[block * BLOCK_LENGTH + k];
                }

                let gradient = self.reducer.nabla(block_input);

                for k in 0..BLOCK_LENGTH {
                    column_gradient[block * BLOCK_LENGTH + k] = gradient[k] * epsilon[block][j];
                }
            }

            next_epsilon.set_column(j, column_gradient);
        }

        (next_epsilon, ())
    }

    fn regularize(&self) -> Self::Delta {
        ()
    }

    fn apply_gradient(&mut self, _gradient: &Self::Delta) {
        // Noop
    }
}
