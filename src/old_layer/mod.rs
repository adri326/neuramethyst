mod dense;
pub use dense::NeuraDenseLayer;

mod convolution;
pub use convolution::{NeuraConv1DPadLayer, NeuraConv2DBlockLayer, NeuraConv2DPadLayer};

mod dropout;
pub use dropout::NeuraDropoutLayer;

mod softmax;
pub use softmax::NeuraSoftmaxLayer;

mod one_hot;
pub use one_hot::NeuraOneHotLayer;

mod lock;
pub use lock::NeuraLockLayer;

mod pool;
pub use pool::{NeuraGlobalPoolLayer, NeuraPool1DLayer};

mod reshape;
pub use reshape::{NeuraFlattenLayer, NeuraReshapeLayer};

use crate::algebra::NeuraVectorSpace;

pub trait NeuraLayer {
    type Input;
    type Output;

    fn eval(&self, input: &Self::Input) -> Self::Output;
}

pub trait NeuraTrainableLayer: NeuraLayer {
    /// The representation of the layer gradient as a vector space
    type Delta: NeuraVectorSpace;

    /// Computes the backpropagation term and the derivative of the internal weights,
    /// using the `input` vector outputted by the previous layer and the backpropagation term `epsilon` of the next layer.
    ///
    /// Note: we introduce the term `epsilon`, which together with the activation of the current function can be used to compute `delta_l`:
    /// ```no_rust
    /// f_l'(a_l) * epsilon_l = delta_l
    /// ```
    ///
    /// The function should then return a pair `(epsilon_{l-1}, δW_l)`,
    /// with `epsilon_{l-1}` being multiplied by `f_{l-1}'(activation)` by the next layer to obtain `delta_{l-1}`.
    /// Using this intermediate value for `delta` allows us to isolate it computation to the respective layers.
    fn backpropagate(
        &self,
        input: &Self::Input,
        epsilon: Self::Output,
    ) -> (Self::Input, Self::Delta);

    /// Computes the regularization
    fn regularize(&self) -> Self::Delta;

    /// Applies `δW_l` to the weights of the layer
    fn apply_gradient(&mut self, gradient: &Self::Delta);

    /// Called before an iteration begins, to allow the layer to set itself up for training.
    #[inline(always)]
    fn prepare_epoch(&mut self) {}

    /// Called at the end of training, to allow the layer to clean itself up
    #[inline(always)]
    fn cleanup(&mut self) {}
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

    ( "one_hot" ) => {
        $crate::layer::NeuraOneHotLayer as $crate::layer::NeuraOneHotLayer<2, _>
    };

    ( "lock", $layer:expr ) => {
        $crate::layer::NeuraLockLayer($layer)
    };

    ( "conv1d_pad", $length:expr, $feats:expr; $window:expr; $layer:expr ) => {
        $crate::layer::NeuraConv1DPadLayer::new($layer, Default::default()) as $crate::layer::NeuraConv1DPadLayer<$length, $feats, $window, _>
    };

    ( "conv1d_pad"; $window:expr; $layer:expr ) => {
        $crate::layer::NeuraConv1DPadLayer::new($layer, Default::default()) as $crate::layer::NeuraConv1DPadLayer<_, _, $window, _>
    };

    ( "conv2d_pad", $feats:expr, $length:expr; $width:expr, $window:expr; $layer:expr ) => {
        $crate::layer::NeuraConv2DPadLayer::new($layer, Default::default(), $width) as $crate::layer::NeuraConv2DPadLayer<$length, $feats, $window, _>
    };

    ( "conv2d_pad"; $width:expr, $window:expr; $layer:expr ) => {
        $crate::layer::NeuraConv2DPadLayer::new($layer, Default::default(), $width) as $crate::layer::NeuraConv2DPadLayer<_, _, $window, _>
    };

    ( "conv2d_block", $feats:expr, $width:expr, $height:expr; $block_size:expr; $layer:expr ) => {
        $crate::layer::NeuraConv2DBlockLayer::new($layer) as $crate::layer::NeuraConv2DBlockLayer<$width, $height, $feats, $block_size, _>
    };

    ( "conv2d_block", $width:expr, $height:expr; $block_size:expr; $layer:expr ) => {
        $crate::layer::NeuraConv2DBlockLayer::new($layer) as $crate::layer::NeuraConv2DBlockLayer<$width, $height, _, $block_size, _>
    };

    ( "pool_global"; $reduce:expr ) => {
        $crate::layer::NeuraGlobalPoolLayer::new($reduce) as $crate::layer::NeuraGlobalPoolLayer<_, _, _>
    };

    ( "pool_global", $feats:expr, $length:expr; $reduce:expr ) => {
        $crate::layer::NeuraGlobalPoolLayer::new($reduce) as $crate::layer::NeuraGlobalPoolLayer<$length, $feats, _>
    };

    ( "pool1d", $blocklength:expr; $reduce:expr ) => {
        $crate::layer::NeuraPool1DLayer::new($reduce) as $crate::layer::NeuraPool1DLayer<_, $blocklength, _, _>
    };

    ( "pool1d", $blocks:expr, $blocklength:expr; $reduce:expr ) => {
        $crate::layer::NeuraPool1DLayer::new($reduce) as $crate::layer::NeuraPool1DLayer<$blocks, $blocklength, _, _>
    };

    ( "pool1d", $feats:expr, $blocks:expr, $blocklength:expr; $reduce:expr ) => {
        $crate::layer::NeuraPool1DLayer::new($reduce) as $crate::layer::NeuraPool1DLayer<$blocks, $blocklength, $feats, _>
    };

    ( "unstable_flatten" ) => {
        $crate::layer::NeuraFlattenLayer::new() as $crate::layer::NeuraFlattenLayer<_, _, f64>
    };

    ( "unstable_flatten", $width:expr, $height:expr ) => {
        $crate::layer::NeuraFlattenLayer::new() as $crate::layer::NeuraFlattenLayer<$width, $height, f64>
    };

    ( "unstable_reshape", $height:expr ) => {
        $crate::layer::NeuraReshapeLayer::new() as $crate::layer::NeuraReshapeLayer<_, $height, f64>
    };

    ( "unstable_reshape", $width:expr, $height:expr ) => {
        $crate::layer::NeuraReshapeLayer::new() as $crate::layer::NeuraReshapeLayer<$width, $height, f64>
    };
}
