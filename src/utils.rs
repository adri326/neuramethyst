use nalgebra::DVector;

#[allow(dead_code)]
pub(crate) fn assign_add_vector<const N: usize>(sum: &mut [f64; N], operand: &[f64; N]) {
    for i in 0..N {
        sum[i] += operand[i];
    }
}

struct Chunked<J: Iterator> {
    iter: J,
    chunk_size: usize,
}

impl<J: Iterator> Iterator for Chunked<J> {
    type Item = Vec<J::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut result = Vec::with_capacity(self.chunk_size);

        for _ in 0..self.chunk_size {
            if let Some(item) = self.iter.next() {
                result.push(item);
            } else {
                break;
            }
        }

        if !result.is_empty() {
            Some(result)
        } else {
            None
        }
    }
}

struct ShuffleCycled<I: Iterator, R: rand::Rng> {
    buffer: Vec<I::Item>,
    index: usize,
    iter: I,
    rng: R,
}

impl<I: Iterator, R: rand::Rng> Iterator for ShuffleCycled<I, R>
where
    I::Item: Clone,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        use rand::prelude::SliceRandom;

        if let Some(next) = self.iter.next() {
            // Base iterator is not empty yet
            self.buffer.push(next.clone());
            Some(next)
        } else if !self.buffer.is_empty() {
            if self.index == 0 {
                // Shuffle the vector and return the first element, setting the index to 1
                self.buffer.shuffle(&mut self.rng);
                self.index = 1;
                Some(self.buffer[0].clone())
            } else {
                // Keep consuming the shuffled vector
                let res = self.buffer[self.index].clone();
                self.index = (self.index + 1) % self.buffer.len();
                Some(res)
            }
        } else {
            None
        }
    }
}

pub fn cycle_shuffling<I: Iterator>(iter: I, rng: impl rand::Rng) -> impl Iterator<Item = I::Item>
where
    I::Item: Clone,
{
    let size_hint = iter.size_hint();
    let size_hint = size_hint.1.unwrap_or(size_hint.0).max(1);

    ShuffleCycled {
        buffer: Vec::with_capacity(size_hint),
        index: 0,
        iter,
        rng,
    }
}

#[cfg(test)]
pub(crate) fn uniform_vector(length: usize) -> nalgebra::DVector<f64> {
    use rand::Rng;

    let mut rng = rand::thread_rng();
    DVector::from_fn(length, |_, _| -> f64 { rng.gen() })
}

pub fn one_hot(value: usize, categories: usize) -> DVector<f32> {
    let mut res = DVector::from_element(categories, 0.0);
    if value < categories {
        res[value] = 1.0;
    }
    res
}

pub fn argmax<F: PartialOrd>(array: &[F]) -> usize {
    let mut res = 0;

    for n in 1..array.len() {
        if array[n] > array[res] {
            res = n;
        }
    }

    res
}

#[cfg(test)]
#[macro_export]
macro_rules! assert_approx {
    ( $left:expr, $right:expr, $epsilon:expr ) => {
        let left = $left;
        let right = $right;
        if ((left - right) as f64).abs() >= $epsilon as f64 {
            panic!("Expected {} to be approximately equal to {}", left, right);
        }
    };
}

// TODO: put this behind a feature
pub fn plot_losses(losses: Vec<(f64, f64)>, width: u32, height: u32) {
    use textplots::{Chart, ColorPlot, Plot, Shape};

    let train_losses: Vec<_> = losses
        .iter()
        .enumerate()
        .map(|(x, y)| (x as f32, y.0 as f32))
        .collect();
    let val_losses: Vec<_> = losses
        .iter()
        .enumerate()
        .map(|(x, y)| (x as f32, y.1 as f32))
        .collect();

    Chart::new(width, height, 0.0, losses.len() as f32)
        .lineplot(&Shape::Lines(&train_losses))
        .linecolorplot(&Shape::Lines(&val_losses), (255, 0, 255).into())
        .nice();
}

pub(crate) fn unwrap_or_clone<T: Clone>(value: std::rc::Rc<T>) -> T {
    // TODO: replace with Rc::unwrap_or_clone once https://github.com/rust-lang/rust/issues/93610 is closed
    std::rc::Rc::try_unwrap(value).unwrap_or_else(|value| (*value).clone())
}

#[cfg(feature = "visualization")]
pub fn draw_neuron_activation<F: Fn([f32; 2]) -> Vec<f32>>(
    callback: F,
    scale: f32,
    width: u32,
    height: u32,
) {
    use viuer::Config;

    let mut image = image::RgbImage::new(width, height);

    fn sigmoid(x: f32) -> f32 {
        1.9 / (1.0 + (-x * 3.0).exp()) - 0.9
    }

    for y in 0..height {
        let y2 = 2.0 * y as f32 / height as f32 - 1.0;
        for x in 0..width {
            let x2 = 2.0 * x as f32 / width as f32 - 1.0;
            let activation = callback([x2 * scale, y2 * scale]);
            let r = (sigmoid(activation.get(0).copied().unwrap_or(-1.0)) * 255.0).floor() as u8;
            let g = (sigmoid(activation.get(1).copied().unwrap_or(-1.0)) * 255.0).floor() as u8;
            let b = (sigmoid(activation.get(2).copied().unwrap_or(-1.0)) * 255.0).floor() as u8;

            *image.get_pixel_mut(x, y) = image::Rgb([r, g, b]);
        }
    }

    let config = Config {
        use_kitty: false,
        truecolor: true,
        // absolute_offset: false,
        ..Default::default()
    };

    viuer::print(&image::DynamicImage::ImageRgb8(image), &config).unwrap();
}
