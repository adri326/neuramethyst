pub mod activation;
pub mod loss;
pub mod regularize;

pub trait NeuraDerivable<F> {
    fn eval(&self, input: F) -> F;

    /// Should return the derivative of `self.eval(input)`
    fn derivate(&self, at: F) -> F;

    /// Should return a hint for how much the variance for a random initialization should be
    #[inline(always)]
    fn variance_hint(&self) -> f64 {
        1.0
    }

    /// Should return a hint for what the default bias value should be
    #[inline(always)]
    fn bias_hint(&self) -> f64 {
        0.0
    }
}

pub trait NeuraLoss {
    type Input;
    type Target;

    fn eval(&self, target: &Self::Target, actual: &Self::Input) -> f64;

    /// Should return the gradient of the loss function according to `actual`
    /// ($\nabla_{\texttt{actual}} \texttt{self.eval}(\texttt{target}, \texttt{actual})$).
    fn nabla(&self, target: &Self::Target, actual: &Self::Input) -> Self::Input;
}
