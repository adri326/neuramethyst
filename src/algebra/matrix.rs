use std::borrow::Borrow;

use super::*;
use boxed_array::from_cloned;
use num::Float;

/// A simple abstraction around `[[F; WIDTH]; HEIGHT]`,
/// which ensures that all allocations that depend on `WIDTH` or `HEIGHT` are done on the heap,
/// without losing the length information.
#[derive(Clone, Debug, PartialEq)]
pub struct NeuraMatrix<const WIDTH: usize, const HEIGHT: usize, F> {
    pub data: Box<[[F; WIDTH]; HEIGHT]>,
}

impl<const WIDTH: usize, const HEIGHT: usize, F> NeuraMatrix<WIDTH, HEIGHT, F> {
    #[inline(always)]
    pub fn from_value(value: F) -> Self
    where
        F: Clone,
    {
        Self {
            data: from_cloned(&value),
        }
    }

    #[inline(always)]
    pub fn get(&self, x: usize, y: usize) -> Option<&F> {
        if x >= WIDTH || y >= HEIGHT {
            return None;
        }

        Some(&self.data[y][x])
    }

    #[inline]
    pub fn set_row(&mut self, y: usize, row: impl Borrow<[F; WIDTH]>)
    where
        F: Clone,
    {
        if y >= HEIGHT {
            panic!(
                "Cannot set row {} of NeuraMatrix<{}, {}, _>: row index out of bound",
                y, WIDTH, HEIGHT
            );
        }

        let row = row.borrow();
        for j in 0..WIDTH {
            self.data[y][j] = row[j].clone();
        }
    }
}

impl<const WIDTH: usize, const HEIGHT: usize, F: Float> NeuraMatrix<WIDTH, HEIGHT, F> {
    /// Returns `self * vector`
    pub fn multiply_vector(&self, vector: impl Borrow<[F; WIDTH]>) -> NeuraVector<HEIGHT, F> {
        let mut result: NeuraVector<HEIGHT, F> = NeuraVector::from_value(F::zero());
        let vector = vector.borrow();

        for i in 0..HEIGHT {
            let mut sum = F::zero();
            for k in 0..WIDTH {
                sum = sum + self.data[i][k] * vector[k];
            }
            result[i] = sum;
        }

        result
    }

    /// Returns `transpose(self) * vector`,
    /// without actually performing the transpose operation
    pub fn transpose_multiply_vector(
        &self,
        vector: impl AsRef<[F; HEIGHT]>,
    ) -> NeuraVector<WIDTH, F> {
        let mut result: NeuraVector<WIDTH, F> = NeuraVector::from_value(F::zero());
        let vector = vector.as_ref();

        for j in 0..WIDTH {
            let mut sum = F::zero();
            for k in 0..HEIGHT {
                sum = sum + self.data[k][j] * vector[k];
            }
            result[j] = sum;
        }

        result
    }
}

impl<const LENGTH: usize, F: Default + Clone> NeuraMatrix<LENGTH, LENGTH, F> {
    pub fn from_diagonal(vector: impl AsRef<[F; LENGTH]>) -> Self {
        let mut result: NeuraMatrix<LENGTH, LENGTH, F> = NeuraMatrix::default();
        let vector = vector.as_ref();

        for i in 0..LENGTH {
            result[i][i] = vector[i].clone();
        }

        result
    }
}

impl<const WIDTH: usize, const HEIGHT: usize, F: Float + From<f64> + Into<f64>> NeuraVectorSpace
    for NeuraMatrix<WIDTH, HEIGHT, F>
{
    fn add_assign(&mut self, other: &Self) {
        for i in 0..HEIGHT {
            for j in 0..WIDTH {
                self.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
    }

    fn mul_assign(&mut self, by: f64) {
        let by: F = by.into();
        for i in 0..HEIGHT {
            for j in 0..WIDTH {
                self.data[i][j] = self.data[i][j] * by;
            }
        }
    }

    #[inline(always)]
    fn zero() -> Self {
        Self::from_value(F::zero())
    }

    fn norm_squared(&self) -> f64 {
        let mut sum = F::zero();

        for i in 0..HEIGHT {
            for j in 0..WIDTH {
                let x = self.data[i][j];
                sum = sum + x * x;
            }
        }

        sum.into()
    }
}

impl<const WIDTH: usize, const HEIGHT: usize, F> From<Box<[[F; WIDTH]; HEIGHT]>>
    for NeuraMatrix<WIDTH, HEIGHT, F>
{
    #[inline]
    fn from(data: Box<[[F; WIDTH]; HEIGHT]>) -> Self {
        Self { data }
    }
}

impl<const WIDTH: usize, const HEIGHT: usize, F> From<NeuraMatrix<WIDTH, HEIGHT, F>>
    for Box<[[F; WIDTH]; HEIGHT]>
{
    #[inline]
    fn from(matrix: NeuraMatrix<WIDTH, HEIGHT, F>) -> Self {
        matrix.data
    }
}

impl<const WIDTH: usize, const HEIGHT: usize, F: Default + Clone> From<&[[F; WIDTH]; HEIGHT]>
    for NeuraMatrix<WIDTH, HEIGHT, F>
{
    /// **Warning:** when using this function, make sure that the array is not allocated on the stack
    /// or that `WIDTH` and `HEIGHT` are bounded.
    #[inline]
    fn from(data: &[[F; WIDTH]; HEIGHT]) -> Self {
        let mut res = Self::default();

        for i in 0..HEIGHT {
            for j in 0..WIDTH {
                res[i][j] = data[i][j].clone();
            }
        }

        res
    }
}

impl<const WIDTH: usize, const HEIGHT: usize, F> From<[[F; WIDTH]; HEIGHT]>
    for NeuraMatrix<WIDTH, HEIGHT, F>
{
    /// **Warning:** when using this function, make sure that `WIDTH` and `HEIGHT` are bounded.
    fn from(data: [[F; WIDTH]; HEIGHT]) -> Self {
        Self {
            data: Box::new(data),
        }
    }
}

impl<const WIDTH: usize, const HEIGHT: usize, F> std::ops::Index<(usize, usize)>
    for NeuraMatrix<WIDTH, HEIGHT, F>
{
    type Output = F;

    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        if index.0 >= WIDTH || index.1 >= HEIGHT {
            panic!(
                "Index out of bound: tried indexing matrix element ({}, {}), which is outside of NeuraMatrix<{}, {}, _>",
                index.0, index.1, WIDTH, HEIGHT
            );
        }

        &self.data[index.1][index.0]
    }
}

impl<const WIDTH: usize, const HEIGHT: usize, F> std::ops::IndexMut<(usize, usize)>
    for NeuraMatrix<WIDTH, HEIGHT, F>
{
    #[inline]
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        if index.0 >= WIDTH || index.1 >= HEIGHT {
            panic!(
                "Index out of bound: tried indexing matrix element ({}, {}), which is outside of NeuraMatrix<{}, {}, _>",
                index.0, index.1, WIDTH, HEIGHT
            );
        }

        &mut self.data[index.1][index.0]
    }
}

impl<const WIDTH: usize, const HEIGHT: usize, F> std::ops::Index<usize>
    for NeuraMatrix<WIDTH, HEIGHT, F>
{
    type Output = [F; WIDTH];

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        if index >= HEIGHT {
            panic!(
                "Index out of bound: tried indexing matrix row {}, which is outside of NeuraMatrix<{}, {}, _>",
                index, WIDTH, HEIGHT
            );
        }

        &self.data[index]
    }
}

impl<const WIDTH: usize, const HEIGHT: usize, F> std::ops::IndexMut<usize>
    for NeuraMatrix<WIDTH, HEIGHT, F>
{
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index >= HEIGHT {
            panic!(
                "Index out of bound: tried indexing matrix row {}, which is outside of NeuraMatrix<{}, {}, _>",
                index, WIDTH, HEIGHT
            );
        }

        &mut self.data[index]
    }
}

impl<const WIDTH: usize, const HEIGHT: usize, F> AsRef<[[F; WIDTH]; HEIGHT]>
    for NeuraMatrix<WIDTH, HEIGHT, F>
{
    #[inline(always)]
    fn as_ref(&self) -> &[[F; WIDTH]; HEIGHT] {
        &self.data
    }
}

impl<const WIDTH: usize, const HEIGHT: usize, F> Borrow<[[F; WIDTH]; HEIGHT]>
    for NeuraMatrix<WIDTH, HEIGHT, F>
{
    #[inline(always)]
    fn borrow(&self) -> &[[F; WIDTH]; HEIGHT] {
        &self.data
    }
}

impl<const WIDTH: usize, const HEIGHT: usize, F: Default + Clone> Default
    for NeuraMatrix<WIDTH, HEIGHT, F>
{
    #[inline(always)]
    fn default() -> Self {
        Self::from_value(F::default())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_index() {
        let mut matrix: NeuraMatrix<1000, 1000, f64> = NeuraMatrix::from_value(0.0);

        matrix[100][200] = 0.3;
        assert_eq!(matrix[(200, 100)], 0.3);
        matrix[(999, 999)] = 0.5;
        assert_eq!(matrix[999][999], 0.5);
    }

    #[test]
    #[should_panic(
        expected = "Index out of bound: tried indexing matrix row 100, which is outside of NeuraMatrix<100, 100, _>"
    )]
    fn test_index_oob() {
        let matrix: NeuraMatrix<100, 100, f64> = NeuraMatrix::from_value(0.0);

        let _ = matrix[100];
    }
}
