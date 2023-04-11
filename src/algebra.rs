/// An extension of `std::ops::AddAssign` and `std::ops::Default`
pub trait NeuraAddAssign {
    fn add_assign(&mut self, other: &Self);

    fn default() -> Self;
}

impl<Left: NeuraAddAssign, Right: NeuraAddAssign> NeuraAddAssign for (Left, Right) {
    fn add_assign(&mut self, other: &Self) {
        NeuraAddAssign::add_assign(&mut self.0, &other.0);
        NeuraAddAssign::add_assign(&mut self.1, &other.1);
    }

    fn default() -> Self {
        (Left::default(), Right::default())
    }
}

impl<const N: usize, T: NeuraAddAssign + Clone> NeuraAddAssign for [T; N] {
    fn add_assign(&mut self, other: &[T; N]) {
        for i in 0..N {
            NeuraAddAssign::add_assign(&mut self[i], &other[i]);
        }
    }

    fn default() -> Self {
        let mut res: Vec<T> = Vec::with_capacity(N);

        for _ in 0..N {
            res.push(T::default());
        }

        res.try_into().unwrap_or_else(|_| {
            // TODO: check that this panic is optimized away
            unreachable!()
        })
    }
}

macro_rules! base {
    ( $type:ty ) => {
        impl NeuraAddAssign for $type {
            fn add_assign(&mut self, other: &Self) {
                std::ops::AddAssign::add_assign(self, other);
            }

            fn default() -> Self {
                <Self as Default>::default()
            }
        }
    }
}

base!(f32);
base!(f64);
