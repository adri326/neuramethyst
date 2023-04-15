mod dense;
pub use dense::NeuraDenseLayer;

mod dropout;
pub use dropout::NeuraDropoutLayer;

mod softmax;
pub use softmax::NeuraSoftmaxLayer;

pub trait NeuraLayer {
    type Input;
    type Output;

    fn eval(&self, input: &Self::Input) -> Self::Output;
}

#[macro_export]
macro_rules! neura_layer {
    ( "dense", $( $shape:expr ),*; $activation:expr ) => {
        $crate::layer::NeuraDenseLayer::from_rng(&mut rand::thread_rng(), $activation, $crate::derivable::regularize::NeuraL0)
            as neura_layer!("_dense_shape", $($shape),*)
    };

    ( "dense", $( $shape:expr ),*; $activation:expr, $regularization:expr ) => {
        $crate::layer::NeuraDenseLayer::from_rng(&mut rand::thread_rng(), $activation, $regularization)
            as neura_layer!("_dense_shape", $($shape),*)
    };

    ( "_dense_shape", $output:expr ) => {
        $crate::layer::NeuraDenseLayer<_, _, _, $output>
    };

    ( "_dense_shape", $input:expr, $output:expr ) => {
        $crate::layer::NeuraDenseLayer<_, _, $input, $output>
    };

    ( "dropout", $probability:expr ) => {
        $crate::layer::NeuraDropoutLayer::new($probability, rand::thread_rng())
            as $crate::layer::NeuraDropoutLayer<_, _>
    };

    ( "softmax" ) => {
        $crate::layer::NeuraSoftmaxLayer::new() as $crate::layer::NeuraSoftmaxLayer<_>
    };

    ( "softmax", $length:expr ) => {
        $crate::layer::NeuraSoftmaxLayer::new() as $crate::layer::NeuraSoftmaxLayer<$length>
    };
}
