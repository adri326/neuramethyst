/// An extension of `std::ops::AddAssign`
pub trait NeuraAddAssign {
    fn add_assign(&mut self, other: &Self);
}

impl<Left: NeuraAddAssign, Right: NeuraAddAssign> NeuraAddAssign for (Left, Right) {
    fn add_assign(&mut self, other: &Self) {
        NeuraAddAssign::add_assign(&mut self.0, &other.0);
        NeuraAddAssign::add_assign(&mut self.1, &other.1);
    }
}

impl<const N: usize, T: NeuraAddAssign> NeuraAddAssign for [T; N] {
    fn add_assign(&mut self, other: &[T; N]) {
        for i in 0..N {
            NeuraAddAssign::add_assign(&mut self[i], &other[i]);
        }
    }
}

macro_rules! base {
    ( $type:ty ) => {
        impl NeuraAddAssign for $type {
            fn add_assign(&mut self, other: &Self) {
                std::ops::AddAssign::add_assign(self, other);
            }
        }
    }
}

base!(f32);
base!(f64);
