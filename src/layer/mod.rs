mod dense;
pub use dense::NeuraDenseLayer;

mod dropout;
pub use dropout::NeuraDropoutLayer;

pub trait NeuraLayer {
    type Input;
    type Output;

    fn eval(&self, input: &Self::Input) -> Self::Output;
}

#[macro_export]
macro_rules! neura_layer {
    ( "dense", $activation:expr, $output:expr ) => {
        NeuraDenseLayer::from_rng(&mut rand::thread_rng(), $activation)
            as NeuraDenseLayer<_, _, $output>
    };

    ( "dense", $activation:expr, $output:expr, $input:expr ) => {
        NeuraDenseLayer::from_rng(&mut rand::thread_rng(), $activation)
            as NeuraDenseLayer<_, $input, $output>
    };

    ( "dropout", $probability:expr ) => {
        NeuraDropoutLayer::new($probability, rand::thread_rng())
            as NeuraDropoutLayer<_, _>
    };
}
