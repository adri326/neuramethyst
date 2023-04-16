use crate::algebra::{NeuraMatrix, NeuraVector};

use super::*;

/// A 1-dimensional convolutional
#[derive(Clone, Debug)]
pub struct NeuraConv1DPad<
    const LENGTH: usize,
    const IN_FEATS: usize,
    const WINDOW: usize,
    Layer: NeuraLayer<Input = NeuraVector<{ IN_FEATS * WINDOW }, f64>>,
> {
    inner_layer: Layer,
    pad_with: NeuraVector<IN_FEATS, f64>,
}

impl<
        const LENGTH: usize,
        const IN_FEATS: usize,
        const WINDOW: usize,
        Layer: NeuraLayer<Input = NeuraVector<{ IN_FEATS * WINDOW }, f64>>,
    > NeuraConv1DPad<LENGTH, IN_FEATS, WINDOW, Layer>
where
    [u8; IN_FEATS * WINDOW]: Sized,
{
    pub fn new(inner_layer: Layer, pad_with: NeuraVector<IN_FEATS, f64>) -> Self {
        Self {
            inner_layer,
            pad_with,
        }
    }

    fn iterate_windows<'a>(
        &'a self,
        input: &'a NeuraMatrix<IN_FEATS, LENGTH, f64>,
    ) -> impl Iterator<Item = (usize, Layer::Input)> + 'a {
        (0..LENGTH).map(move |window_center| {
            let mut virtual_input: NeuraVector<{ IN_FEATS * WINDOW }, f64> = NeuraVector::default();
            for i in 0..WINDOW {
                let input_index = i as isize + window_center as isize - (WINDOW - 1) as isize / 2;

                if input_index < 0 || input_index >= LENGTH as isize {
                    for j in 0..IN_FEATS {
                        virtual_input[i * IN_FEATS + j] = self.pad_with[j];
                    }
                } else {
                    for j in 0..IN_FEATS {
                        virtual_input[i * IN_FEATS + j] = input[input_index as usize][j];
                    }
                }
            }

            (window_center, virtual_input)
        })
    }
}

impl<
        const LENGTH: usize,
        const IN_FEATS: usize,
        const OUT_FEATS: usize,
        const WINDOW: usize,
        Layer: NeuraLayer<
            Input = NeuraVector<{ IN_FEATS * WINDOW }, f64>,
            Output = NeuraVector<OUT_FEATS, f64>,
        >,
    > NeuraLayer for NeuraConv1DPad<LENGTH, IN_FEATS, WINDOW, Layer>
{
    type Input = NeuraMatrix<IN_FEATS, LENGTH, f64>;
    type Output = NeuraMatrix<OUT_FEATS, LENGTH, f64>;

    fn eval(&self, input: &Self::Input) -> Self::Output {
        let mut res = NeuraMatrix::default();

        for (window_center, virtual_input) in self.iterate_windows(input) {
            res.set_row(window_center, self.inner_layer.eval(&virtual_input));
        }

        res
    }
}

impl<
        const LENGTH: usize,
        const IN_FEATS: usize,
        const OUT_FEATS: usize,
        const WINDOW: usize,
        Layer: NeuraLayer<
            Input = NeuraVector<{ IN_FEATS * WINDOW }, f64>,
            Output = NeuraVector<OUT_FEATS, f64>,
        >,
    > NeuraTrainableLayer for NeuraConv1DPad<LENGTH, IN_FEATS, WINDOW, Layer>
where
    Layer: NeuraTrainableLayer,
{
    type Delta = <Layer as NeuraTrainableLayer>::Delta;

    fn backpropagate(
        &self,
        input: &Self::Input,
        epsilon: Self::Output,
    ) -> (Self::Input, Self::Delta) {
        let mut next_epsilon = Self::Input::default();
        let mut weights_gradient_sum = Self::Delta::zero();

        // TODO: consume epsilon efficiently
        for (window_center, virtual_input) in self.iterate_windows(input) {
            let epsilon = NeuraVector::from(&epsilon[window_center]);

            let (layer_next_epsilon, weights_gradient) =
                self.inner_layer.backpropagate(&virtual_input, epsilon);
            weights_gradient_sum.add_assign(&weights_gradient);

            for i in 0..WINDOW {
                // Re-compute the positions in `input` matching the positions in `layer_next_epsilon` and `virtual_input`
                let input_index = window_center as isize + i as isize - (WINDOW - 1) as isize / 2;
                if input_index < 0 || input_index >= LENGTH as isize {
                    continue;
                }
                let input_index = input_index as usize;

                for j in 0..IN_FEATS {
                    next_epsilon[input_index][j] += layer_next_epsilon[i * WINDOW + j];
                }
            }
        }

        (next_epsilon, weights_gradient_sum)
    }

    fn regularize(&self) -> Self::Delta {
        self.inner_layer.regularize()
    }

    fn apply_gradient(&mut self, gradient: &Self::Delta) {
        self.inner_layer.apply_gradient(gradient);
    }
}
