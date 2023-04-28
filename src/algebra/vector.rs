use std::borrow::Borrow;

use super::*;
use boxed_array::from_cloned;
use num::Float;

#[derive(Clone, Debug, PartialEq)]
pub struct NeuraVector<const LENGTH: usize, F> {
    pub data: Box<[F; LENGTH]>,
}

impl<const LENGTH: usize, F> NeuraVector<LENGTH, F> {
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
    pub fn get(&self, index: usize) -> Option<&F> {
        if index >= LENGTH {
            None
        } else {
            Some(&self.data[index])
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        LENGTH
    }

    pub fn iter<'a>(&'a self) -> std::slice::Iter<'a, F> {
        self.data.iter()
    }
}

impl<const LENGTH: usize, F: Float> NeuraVector<LENGTH, F> {
    pub fn dot(&self, other: impl AsRef<[F; LENGTH]>) -> F {
        let mut sum = F::zero();
        let other = other.as_ref();

        for i in 0..LENGTH {
            sum = sum + self.data[i] * other[i];
        }

        sum
    }

    /// Returns $left^{\top} \cdot right$, ie. $\ket{left} \bra{right}$
    pub fn reverse_dot<const WIDTH: usize>(
        &self,
        other: impl Borrow<[F; WIDTH]>,
    ) -> NeuraMatrix<WIDTH, LENGTH, F> {
        let mut result: NeuraMatrix<WIDTH, LENGTH, F> = NeuraMatrix::from_value(F::zero());
        let other = other.borrow();

        for i in 0..LENGTH {
            for j in 0..WIDTH {
                result[i][j] = self.data[i] * other[j];
            }
        }

        result
    }

    pub fn hadamard_product(&self, other: impl Borrow<[F; LENGTH]>) -> NeuraVector<LENGTH, F> {
        let mut result: NeuraVector<LENGTH, F> = NeuraVector::from_value(F::zero());
        let other = other.borrow();

        for i in 0..LENGTH {
            result[i] = self.data[i] * other[i];
        }

        result
    }
}

impl<const LENGTH: usize, F: Float + From<f64> + Into<f64>> NeuraVectorSpace
    for NeuraVector<LENGTH, F>
{
    fn add_assign(&mut self, other: &Self) {
        for i in 0..LENGTH {
            self.data[i] = self.data[i] + other.data[i];
        }
    }

    fn mul_assign(&mut self, by: f64) {
        for i in 0..LENGTH {
            self.data[i] = self.data[i] * by.into();
        }
    }

    // #[inline(always)]
    // fn zero() -> Self {
    //     Self::from_value(F::zero())
    // }

    fn norm_squared(&self) -> f64 {
        let mut sum = F::zero();

        for i in 0..LENGTH {
            sum = sum + self.data[i] * self.data[i];
        }

        sum.into()
    }
}

impl<const LENGTH: usize, F> std::ops::Index<usize> for NeuraVector<LENGTH, F> {
    type Output = F;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        if index >= LENGTH {
            panic!(
                "Tried indexing element {} of NeuraVector<{}, _>",
                index, LENGTH
            );
        }

        &self.data[index]
    }
}

impl<const LENGTH: usize, F> std::ops::IndexMut<usize> for NeuraVector<LENGTH, F> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index >= LENGTH {
            panic!(
                "Tried indexing element {} of NeuraVector<{}, _>",
                index, LENGTH
            );
        }

        &mut self.data[index]
    }
}

impl<const LENGTH: usize, F> AsRef<[F; LENGTH]> for NeuraVector<LENGTH, F> {
    #[inline(always)]
    fn as_ref(&self) -> &[F; LENGTH] {
        &self.data
    }
}

impl<const LENGTH: usize, F> AsRef<[F]> for NeuraVector<LENGTH, F> {
    #[inline(always)]
    fn as_ref(&self) -> &[F] {
        self.data.as_ref()
    }
}

impl<const LENGTH: usize, F> Borrow<[F; LENGTH]> for NeuraVector<LENGTH, F> {
    #[inline(always)]
    fn borrow(&self) -> &[F; LENGTH] {
        &self.data
    }
}

impl<const LENGTH: usize, F> Borrow<[F; LENGTH]> for &NeuraVector<LENGTH, F> {
    #[inline(always)]
    fn borrow(&self) -> &[F; LENGTH] {
        &self.data
    }
}

impl<const LENGTH: usize, F> From<Box<[F; LENGTH]>> for NeuraVector<LENGTH, F> {
    fn from(data: Box<[F; LENGTH]>) -> Self {
        Self { data }
    }
}

impl<const LENGTH: usize, F> From<NeuraVector<LENGTH, F>> for Box<[F; LENGTH]> {
    fn from(vector: NeuraVector<LENGTH, F>) -> Self {
        vector.data
    }
}

impl<const LENGTH: usize, F: Default + Clone> From<&[F; LENGTH]> for NeuraVector<LENGTH, F> {
    /// **Warning:** when using this function, make sure that the array is not allocated on the stack,
    /// or that `LENGTH` is bounded.
    fn from(data: &[F; LENGTH]) -> Self {
        let mut res = Self::default();

        for i in 0..LENGTH {
            res.data[i] = data[i].clone();
        }

        res
    }
}

impl<const LENGTH: usize, F> From<[F; LENGTH]> for NeuraVector<LENGTH, F> {
    /// **Warning:** when using this function, make sure that `LENGTH` is bounded.
    fn from(data: [F; LENGTH]) -> Self {
        Self {
            data: Box::new(data),
        }
    }
}

impl<const LENGTH: usize, F: Default + Clone> Default for NeuraVector<LENGTH, F> {
    #[inline(always)]
    fn default() -> Self {
        Self::from_value(F::default())
    }
}

impl<const LENGTH: usize, F> IntoIterator for NeuraVector<LENGTH, F> {
    type Item = F;
    type IntoIter = std::array::IntoIter<F, LENGTH>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, const LENGTH: usize, F> IntoIterator for &'a NeuraVector<LENGTH, F> {
    type Item = &'a F;
    type IntoIter = std::slice::Iter<'a, F>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, const LENGTH: usize, F> IntoIterator for &'a mut NeuraVector<LENGTH, F> {
    type Item = &'a mut F;
    type IntoIter = std::slice::IterMut<'a, F>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

impl<'a, const LENGTH: usize, F: Default + Clone> FromIterator<F> for NeuraVector<LENGTH, F> {
    fn from_iter<T: IntoIterator<Item = F>>(iter: T) -> Self {
        let mut res = Self::default();
        let mut iter = iter.into_iter();

        for i in 0..LENGTH {
            if let Some(next) = iter.next() {
                res[i] = next;
            } else {
                break;
            }
        }

        res
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // #[test]
    // fn test_reverse_dot() {
    //     let left: NeuraVector<_, f64> = [2.0, 3.0, 5.0].into();
    //     let right: NeuraVector<_, f64> = [7.0, 11.0, 13.0, 17.0].into();

    //     let expected: NeuraMatrix<_, _, f64> = [
    //         [14.0, 22.0, 26.0, 34.0],
    //         [21.0, 33.0, 39.0, 51.0],
    //         [35.0, 55.0, 65.0, 85.0],
    //     ]
    //     .into();

    //     let actual = left.reverse_dot(right);

    //     assert_eq!(expected, actual);
    // }
}
