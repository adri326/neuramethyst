mod matrix;
pub use matrix::NeuraMatrix;

mod vector;
pub use vector::NeuraVector;

/// An extension of `std::ops::AddAssign` and `std::ops::Default`
pub trait NeuraVectorSpace {
    fn add_assign(&mut self, other: &Self);

    fn mul_assign(&mut self, by: f64);

    fn zero() -> Self;

    fn norm_squared(&self) -> f64;
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

    #[inline(always)]
    fn zero() -> Self {
        ()
    }

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

    fn zero() -> Self {
        Box::new(T::zero())
    }

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

    fn zero() -> Self {
        (Left::zero(), Right::zero())
    }

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

    fn zero() -> Self {
        let mut res: Vec<T> = Vec::with_capacity(N);

        for _ in 0..N {
            res.push(T::zero());
        }

        res.try_into().unwrap_or_else(|_| {
            // TODO: check that this panic is optimized away
            unreachable!()
        })
    }

    fn norm_squared(&self) -> f64 {
        self.iter().map(T::norm_squared).sum()
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

            fn zero() -> Self {
                <Self as Default>::default()
            }

            fn norm_squared(&self) -> f64 {
                (self * self) as f64
            }
        }
    };
}

base!(f32);
base!(f64);
