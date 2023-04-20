use crate::algebra::NeuraVector;

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

        if result.len() > 0 {
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
            return Some(next);
        } else if self.buffer.len() > 0 {
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
    use nalgebra::DVector;
    use rand::Rng;

    let mut rng = rand::thread_rng();
    DVector::from_fn(length, |_, _| -> f64 { rng.gen() })
}

pub fn one_hot<const N: usize>(value: usize) -> NeuraVector<N, f64> {
    let mut res = NeuraVector::default();
    if value < N {
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
