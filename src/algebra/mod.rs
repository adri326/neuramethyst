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

    // fn zero() -> Self;

    fn norm_squared(&self) -> f64;
}

pub trait NeuraDynVectorSpace {
    fn add_assign(&mut self, other: &dyn NeuraDynVectorSpace);

    fn mul_assign(&mut self, by: f64);

    fn norm_squared(&self) -> f64;

    /// Trampoline for allowing NeuraDynVectorSpace to be cast back into a known type for add_assign
    fn into_any(&self) -> &dyn Any;
}

impl<T: NeuraVectorSpace + 'static> NeuraDynVectorSpace for T {
    fn add_assign(&mut self, other: &dyn NeuraDynVectorSpace) {
        let Some(other) = other.into_any().downcast_ref::<Self>() else {
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

    // #[inline(always)]
    // fn zero() -> Self {
    //     ()
    // }

    fn norm_squared(&self) -> f64 {
        0.0
    }
}

impl<T: NeuraVectorSpace> NeuraVectorSpace for Box<T> {
    fn add_assign(&mut self, other: &Self) {
        self.as_mut().add_assign(other.as_ref());
    }

    fn mul_assign(&mut self, by: f64) {
        self.as_mut().mul_assign(by);
    }

    // fn zero() -> Self {
    //     Box::new(T::zero())
    // }

    fn norm_squared(&self) -> f64 {
        self.as_ref().norm_squared()
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

    // fn zero() -> Self {
    //     let mut res: Vec<T> = Vec::with_capacity(N);

    //     for _ in 0..N {
    //         res.push(T::zero());
    //     }

    //     res.try_into().unwrap_or_else(|_| {
    //         // TODO: check that this panic is optimized away
    //         unreachable!()
    //     })
    // }

    fn norm_squared(&self) -> f64 {
        self.iter().map(T::norm_squared).sum()
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
