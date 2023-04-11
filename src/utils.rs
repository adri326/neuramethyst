pub fn multiply_matrix_vector<const WIDTH: usize, const HEIGHT: usize>(
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
