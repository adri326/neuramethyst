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

pub(crate) fn chunked<I: Iterator>(
    iter: I,
    chunk_size: usize,
) -> impl Iterator<Item = Vec<I::Item>> {
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

    Chunked { iter, chunk_size }
}
