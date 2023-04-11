pub mod activation;
pub mod loss;

pub trait NeuraDerivable<F> {
    fn eval(&self, input: F) -> F;

    /// Should return the derivative of `self.eval(input)`
    fn derivate(&self, at: F) -> F;
}

pub trait NeuraLoss<F> {
    type Out;
    type Target;

    fn eval(&self, target: Self::Target, actual: F) -> Self::Out;

    /// Should return the gradient of the loss function according to `actual`
    /// ($\nabla_{\texttt{actual}} \texttt{self.eval}(\texttt{target}, \texttt{actual})$).
    fn nabla(&self, target: Self::Target, actual: F) -> F;
}
