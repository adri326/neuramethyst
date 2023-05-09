mod matrix;
use std::any::Any;

pub use matrix::NeuraMatrix;

mod vector;
use nalgebra::Matrix;
use num::Float;
pub use vector::NeuraVector;

/// An extension of `std::ops::AddAssign` and `std::ops::Default`
pub trait NeuraVectorSpace {
    fn add_assign(&mut self, other: &Self);

    fn mul_assign(&mut self, by: f64);

    fn norm_squared(&self) -> f64;
}

pub trait NeuraDynVectorSpace: Send {
    fn add_assign(&mut self, other: &dyn NeuraDynVectorSpace);

    fn mul_assign(&mut self, by: f64);

    fn norm_squared(&self) -> f64;

    /// Trampoline for allowing NeuraDynVectorSpace to be cast back into a known type for add_assign
    fn into_any(&self) -> &dyn Any;
}

impl<T: NeuraVectorSpace + Send + 'static> NeuraDynVectorSpace for T {
    fn add_assign(&mut self, other: &dyn NeuraDynVectorSpace) {
        let Some(other) = other.into_any().downcast_ref::<T>() else {
            panic!("Incompatible operand: expected other to be equal to self");
        };

        <Self as NeuraVectorSpace>::add_assign(self, other);
    }

    fn mul_assign(&mut self, by: f64) {
        <Self as NeuraVectorSpace>::mul_assign(self, by);
    }

    fn norm_squared(&self) -> f64 {
        <Self as NeuraVectorSpace>::norm_squared(self)
    }

    fn into_any(&self) -> &dyn Any {
        self
    }
}

impl NeuraVectorSpace for () {
    #[inline(always)]
    fn add_assign(&mut self, _other: &Self) {
        // Noop
    }

    #[inline(always)]
    fn mul_assign(&mut self, _by: f64) {
        // Noop
    }

    fn norm_squared(&self) -> f64 {
        0.0
    }
}

impl<T: NeuraVectorSpace + ?Sized> NeuraVectorSpace for Box<T> {
    fn add_assign(&mut self, other: &Self) {
        self.as_mut().add_assign(other.as_ref());
    }

    fn mul_assign(&mut self, by: f64) {
        self.as_mut().mul_assign(by);
    }

    fn norm_squared(&self) -> f64 {
        self.as_ref().norm_squared()
    }
}

impl NeuraVectorSpace for dyn NeuraDynVectorSpace {
    fn add_assign(&mut self, other: &Self) {
        <dyn NeuraDynVectorSpace>::add_assign(self, &*other)
    }

    fn mul_assign(&mut self, by: f64) {
        <dyn NeuraDynVectorSpace>::mul_assign(self, by)
    }

    fn norm_squared(&self) -> f64 {
        <dyn NeuraDynVectorSpace>::norm_squared(self)
    }
}

impl<Left: NeuraVectorSpace, Right: NeuraVectorSpace> NeuraVectorSpace for (Left, Right) {
    fn add_assign(&mut self, other: &Self) {
        NeuraVectorSpace::add_assign(&mut self.0, &other.0);
        NeuraVectorSpace::add_assign(&mut self.1, &other.1);
    }

    fn mul_assign(&mut self, by: f64) {
        NeuraVectorSpace::mul_assign(&mut self.0, by);
        NeuraVectorSpace::mul_assign(&mut self.1, by);
    }

    // fn zero() -> Self {
    //     (Left::zero(), Right::zero())
    // }

    fn norm_squared(&self) -> f64 {
        self.0.norm_squared() + self.1.norm_squared()
    }
}

impl<const N: usize, T: NeuraVectorSpace + Clone> NeuraVectorSpace for [T; N] {
    fn add_assign(&mut self, other: &[T; N]) {
        for i in 0..N {
            NeuraVectorSpace::add_assign(&mut self[i], &other[i]);
        }
    }

    fn mul_assign(&mut self, by: f64) {
        for i in 0..N {
            NeuraVectorSpace::mul_assign(&mut self[i], by);
        }
    }

    fn norm_squared(&self) -> f64 {
        self.iter().map(T::norm_squared).sum()
    }
}

impl<T: NeuraVectorSpace> NeuraVectorSpace for Vec<T> {
    fn add_assign(&mut self, other: &Self) {
        assert_eq!(self.len(), other.len());

        for (self_item, other_item) in self.iter_mut().zip(other.iter()) {
            self_item.add_assign(other_item);
        }
    }

    fn mul_assign(&mut self, by: f64) {
        for item in self.iter_mut() {
            item.mul_assign(by);
        }
    }

    fn norm_squared(&self) -> f64 {
        let mut res = 0.0;

        for item in self.iter() {
            res += item.norm_squared();
        }

        res
    }
}

impl<F: Float, R: nalgebra::Dim, C: nalgebra::Dim, S: nalgebra::RawStorage<F, R, C>>
    NeuraVectorSpace for Matrix<F, R, C, S>
where
    Matrix<F, R, C, S>: std::ops::MulAssign<F>,
    for<'c> Matrix<F, R, C, S>: std::ops::AddAssign<&'c Matrix<F, R, C, S>>,
{
    fn add_assign(&mut self, other: &Self) {
        *self += other;
    }

    fn mul_assign(&mut self, by: f64) {
        *self *= F::from(by).unwrap();
    }

    fn norm_squared(&self) -> f64 {
        self.iter()
            .map(|x| *x * *x)
            .reduce(|sum, curr| sum + curr)
            .unwrap_or(F::zero())
            .to_f64()
            .unwrap_or(0.0)
    }
}

macro_rules! base {
    ( $type:ty ) => {
        impl NeuraVectorSpace for $type {
            fn add_assign(&mut self, other: &Self) {
                std::ops::AddAssign::add_assign(self, other);
            }

            fn mul_assign(&mut self, other: f64) {
                std::ops::MulAssign::mul_assign(self, other as $type);
            }

            fn norm_squared(&self) -> f64 {
                (self * self) as f64
            }
        }
    };
}

base!(f32);
base!(f64);
