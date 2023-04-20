use crate::algebra::NeuraVectorSpace;

pub mod dense;
pub mod dropout;
pub mod softmax;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum NeuraShape {
    Vector(usize),               // entries
    Matrix(usize, usize),        // rows, columns
    Tensor(usize, usize, usize), // rows, columns, channels
}

impl NeuraShape {
    pub fn size(&self) -> usize {
        match self {
            NeuraShape::Vector(entries) => *entries,
            NeuraShape::Matrix(rows, columns) => rows * columns,
            NeuraShape::Tensor(rows, columns, channels) => rows * columns * channels,
        }
    }
}

pub trait NeuraLayer<Input> {
    type Output;

    fn eval(&self, input: &Input) -> Self::Output;
}

impl<Input: Clone> NeuraLayer<Input> for () {
    type Output = Input;

    #[inline(always)]
    fn eval(&self, input: &Input) -> Self::Output {
        input.clone()
    }
}

pub trait NeuraPartialLayer {
    type Constructed;
    type Err;

    fn construct(self, input_shape: NeuraShape) -> Result<Self::Constructed, Self::Err>;

    fn output_shape(constructed: &Self::Constructed) -> NeuraShape;
}

pub trait NeuraTrainableLayer<Input>: NeuraLayer<Input> {
    /// The representation of the layer gradient as a vector space
    type Gradient: NeuraVectorSpace;

    fn default_gradient(&self) -> Self::Gradient;

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
    fn backprop_layer(&self, input: &Input, epsilon: Self::Output) -> (Input, Self::Gradient);

    /// Computes the regularization
    fn regularize_layer(&self) -> Self::Gradient;

    /// Applies `δW_l` to the weights of the layer
    fn apply_gradient(&mut self, gradient: &Self::Gradient);

    /// Arbitrary computation that can be executed at the start of an epoch
    #[allow(unused_variables)]
    #[inline(always)]
    fn prepare_layer(&mut self, is_training: bool) {}
}

impl<Input: Clone> NeuraTrainableLayer<Input> for () {
    type Gradient = ();

    #[inline(always)]
    fn default_gradient(&self) -> Self::Gradient {
        ()
    }

    #[inline(always)]
    fn backprop_layer(&self, _input: &Input, epsilon: Self::Output) -> (Input, Self::Gradient) {
        (epsilon, ())
    }

    #[inline(always)]
    fn regularize_layer(&self) -> Self::Gradient {
        ()
    }

    #[inline(always)]
    fn apply_gradient(&mut self, _gradient: &Self::Gradient) {
        // Noop
    }
}

/// Temporary implementation of neura_layer
#[macro_export]
macro_rules! neura_layer {
    ( "dense", $output:expr, $type:ty ) => {{
        let res: $crate::layer::dense::NeuraDenseLayerPartial<$type, _, _, _> =
            $crate::layer::dense::NeuraDenseLayer::new_partial(
                $output,
                rand::thread_rng(),
                $crate::derivable::activation::LeakyRelu(0.1),
                $crate::derivable::regularize::NeuraL0,
            );
        res
    }};
    ( "dense", $output:expr ) => {
        $crate::neura_layer!("dense", $output, f32)
    };

    ( "dropout", $probability:expr ) => {
        $crate::layer::dropout::NeuraDropoutLayer::new($probability, rand::thread_rng())
    };

    ( "softmax" ) => {
        $crate::layer::softmax::NeuraSoftmaxLayer::new()
    };
}
