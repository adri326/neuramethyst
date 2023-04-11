mod dense;
pub use dense::NeuraDenseLayer;

pub trait NeuraLayer {
    type Input;
    type Output;

    fn eval(&self, input: &Self::Input) -> Self::Output;
}
