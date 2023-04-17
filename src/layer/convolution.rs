use std::num::NonZeroUsize;

use crate::algebra::{NeuraMatrix, NeuraVector};

use super::*;

/// A 1-dimensional convolutional layer, which operates on rows of the input,
/// and pads them with `self.pad_with`
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct NeuraConv1DPadLayer<
    const LENGTH: usize,
    const IN_FEATS: usize,
    const WINDOW: usize,
    Layer: NeuraLayer<Input = NeuraVector<{ IN_FEATS * WINDOW }, f64>>,
> {
    pub inner_layer: Layer,
    pub pad_with: NeuraVector<IN_FEATS, f64>,
}

impl<
        const LENGTH: usize,
        const IN_FEATS: usize,
        const WINDOW: usize,
        Layer: NeuraLayer<Input = NeuraVector<{ IN_FEATS * WINDOW }, f64>>,
    > NeuraConv1DPadLayer<LENGTH, IN_FEATS, WINDOW, Layer>
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
    > NeuraLayer for NeuraConv1DPadLayer<LENGTH, IN_FEATS, WINDOW, Layer>
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
    > NeuraTrainableLayer for NeuraConv1DPadLayer<LENGTH, IN_FEATS, WINDOW, Layer>
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
                    next_epsilon[input_index][j] += layer_next_epsilon[i * IN_FEATS + j];
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

    fn prepare_epoch(&mut self) {
        self.inner_layer.prepare_epoch();
    }

    fn cleanup(&mut self) {
        self.inner_layer.cleanup();
    }
}

/// Applies and trains a 2d convolution on an image, which has been laid out flat
/// (instead of being a `(width, height, features)` tensor, it should be a `(width * height, features)` matrix).
///
/// The `width` value must be given to the layer, to allow it to interpret the input as a 2D image.
/// This function asserts that `width` divides `LENGTH`.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct NeuraConv2DPadLayer<
    const LENGTH: usize,
    const IN_FEATS: usize,
    const WINDOW: usize,
    Layer: NeuraLayer<Input = NeuraVector<{ IN_FEATS * WINDOW * WINDOW }, f64>>,
> {
    pub inner_layer: Layer,
    pub pad_with: NeuraVector<IN_FEATS, f64>,
    /// The width of the image, in grid units.
    ///
    /// **Class invariant:** `LAYER % width == 0`, `width > 0`
    pub width: NonZeroUsize,
}

impl<
        const LENGTH: usize,
        const IN_FEATS: usize,
        const WINDOW: usize,
        Layer: NeuraLayer<Input = NeuraVector<{ IN_FEATS * WINDOW * WINDOW }, f64>>,
    > NeuraConv2DPadLayer<LENGTH, IN_FEATS, WINDOW, Layer>
{
    pub fn new(inner_layer: Layer, pad_with: NeuraVector<IN_FEATS, f64>, width: usize) -> Self {
        assert!(
            LENGTH % width == 0,
            "width ({}) does not divide LENGTH ({})",
            width,
            LENGTH
        );

        Self {
            inner_layer,
            pad_with,
            width: width.try_into().expect("width cannot be zero!"),
        }
    }

    /// Iterates within the `(WINDOW, WINDOW)` window centered around `x, y`;
    /// Returns a 4-uple `(x' = x + δx, y' = y + δy, δy * WINDOW + δ, y' * width + x')`, with the last element
    /// being set to `None` if `x'` or `y'` are out of bound.
    fn iterate_within_window<'a>(
        &'a self,
        x: usize,
        y: usize,
    ) -> impl Iterator<Item = (isize, isize, usize, Option<usize>)> + 'a {
        let window_offset = (WINDOW as isize - 1) / 2;
        let width = self.width.get() as isize;
        let height = LENGTH as isize / width;

        (0..WINDOW).flat_map(move |dy| {
            let window_index = dy * WINDOW;
            let dy = dy as isize - window_offset;
            (0..WINDOW).map(move |dx| {
                let window_index = window_index + dx;
                let dx = dx as isize - window_offset;

                let x = x as isize + dx;
                let y = y as isize + dy;

                if x < 0 || y < 0 || x >= width || y >= height {
                    (x, y, window_index, None)
                } else {
                    (x, y, window_index, Some((y * width + x) as usize))
                }
            })
        })
    }

    fn iterate_windows<'a>(
        &'a self,
        input: &'a NeuraMatrix<IN_FEATS, LENGTH, f64>,
    ) -> impl Iterator<Item = (usize, usize, Layer::Input)> + 'a {
        (0..self.width.into()).flat_map(move |x| {
            (0..(LENGTH / self.width)).map(move |y| {
                // TODO: hint the compiler that x * y < LENGTH and LENGTH % width == 0?
                let mut virtual_input: Layer::Input = NeuraVector::default();

                for (_, _, window_index, input_index) in self.iterate_within_window(x, y) {
                    if let Some(input_index) = input_index {
                        let input_row: &[f64; IN_FEATS] = &input[input_index];
                        for j in 0..IN_FEATS {
                            virtual_input[window_index * IN_FEATS + j] = input_row[j];
                        }
                    } else {
                        for j in 0..IN_FEATS {
                            virtual_input[window_index * IN_FEATS + j] = self.pad_with[j];
                        }
                    }
                }

                (x, y, virtual_input)
            })
        })
    }
}

impl<
        const LENGTH: usize,
        const IN_FEATS: usize,
        const OUT_FEATS: usize,
        const WINDOW: usize,
        Layer: NeuraLayer<
            Input = NeuraVector<{ IN_FEATS * WINDOW * WINDOW }, f64>,
            Output = NeuraVector<OUT_FEATS, f64>,
        >,
    > NeuraLayer for NeuraConv2DPadLayer<LENGTH, IN_FEATS, WINDOW, Layer>
{
    type Input = NeuraMatrix<IN_FEATS, LENGTH, f64>;
    type Output = NeuraMatrix<OUT_FEATS, LENGTH, f64>;

    fn eval(&self, input: &Self::Input) -> Self::Output {
        let mut res = NeuraMatrix::default();

        for (cx, cy, virtual_input) in self.iterate_windows(input) {
            res.set_row(
                cy * self.width.get() + cx,
                self.inner_layer.eval(&virtual_input),
            );
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
            Input = NeuraVector<{ IN_FEATS * WINDOW * WINDOW }, f64>,
            Output = NeuraVector<OUT_FEATS, f64>,
        >,
    > NeuraTrainableLayer for NeuraConv2DPadLayer<LENGTH, IN_FEATS, WINDOW, Layer>
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

        for (cx, cy, virtual_input) in self.iterate_windows(input) {
            let epsilon = NeuraVector::from(&epsilon[cy * self.width.get() + cx]);

            let (layer_next_epsilon, weights_gradient) =
                self.inner_layer.backpropagate(&virtual_input, epsilon);
            weights_gradient_sum.add_assign(&weights_gradient);

            for (_, _, window_index, input_index) in self.iterate_within_window(cx, cy) {
                if let Some(input_index) = input_index {
                    for j in 0..IN_FEATS {
                        next_epsilon[input_index][j] +=
                            layer_next_epsilon[window_index * IN_FEATS + j];
                    }
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

    fn prepare_epoch(&mut self) {
        self.inner_layer.prepare_epoch();
    }

    fn cleanup(&mut self) {
        self.inner_layer.cleanup();
    }
}
