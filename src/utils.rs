pub(crate) fn multiply_matrix_vector<const WIDTH: usize, const HEIGHT: usize>(
    matrix: &[[f64; WIDTH]; HEIGHT],
    vector: &[f64; WIDTH],
) -> [f64; HEIGHT] {
    let mut result = [0.0; HEIGHT];

    for i in 0..HEIGHT {
        let mut sum = 0.0;
        for k in 0..WIDTH {
            sum += matrix[i][k] * vector[k];
        }
        result[i] = sum;
    }

    result
}

/// Equivalent to `multiply_matrix_vector(transpose(matrix), vector)`.
pub(crate) fn multiply_matrix_transpose_vector<const WIDTH: usize, const HEIGHT: usize>(
    matrix: &[[f64; WIDTH]; HEIGHT],
    vector: &[f64; HEIGHT],
) -> [f64; WIDTH] {
    let mut result = [0.0; WIDTH];

    for i in 0..WIDTH {
        let mut sum = 0.0;
        for k in 0..HEIGHT {
            sum += matrix[k][i] * vector[k];
        }
        result[i] = sum;
    }

    result
}

pub(crate) fn reverse_dot_product<const WIDTH: usize, const HEIGHT: usize>(
    left: &[f64; HEIGHT],
    right: &[f64; WIDTH],
) -> [[f64; WIDTH]; HEIGHT] {
    let mut result = [[0.0; WIDTH]; HEIGHT];

    for i in 0..HEIGHT {
        for j in 0..WIDTH {
            result[i][j] = left[i] * right[j];
        }
    }

    result
}

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

pub(crate) fn chunked<I: Iterator>(
    iter: I,
    chunk_size: usize,
) -> impl Iterator<Item = Vec<I::Item>> {
    Chunked { iter, chunk_size }
}


struct ShuffleCycled<I: Iterator, R: rand::Rng> {
    buffer: Vec<I::Item>,
    index: usize,
    iter: I,
    rng: R,
}

impl<I: Iterator, R: rand::Rng> Iterator for ShuffleCycled<I, R> where I::Item: Clone {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        use rand::prelude::SliceRandom;

        if let Some(next) = self.iter.next() {
            // Base iterator is not empty yet
            self.buffer.push(next.clone());
            return Some(next)
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

pub fn cycle_shuffling<I: Iterator>(
    iter: I,
    rng: impl rand::Rng
) -> impl Iterator<Item=I::Item>
where
    I::Item: Clone
{
    let size_hint = iter.size_hint();
    let size_hint = size_hint.1.unwrap_or(size_hint.0).max(1);

    ShuffleCycled {
        buffer: Vec::with_capacity(size_hint),
        index: 0,
        iter,
        rng
    }
}
