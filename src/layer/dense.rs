use std::marker::PhantomData;

use nalgebra::{DMatrix, DVector};
use num::Float;
use rand::Rng;

use crate::{derivable::NeuraDerivable, err::NeuraDimensionsMismatch};

use super::*;

#[derive(Clone, Debug)]
pub struct NeuraDenseLayer<F: Float, Act: NeuraDerivable<F>, Reg: NeuraDerivable<F>> {
    pub weights: DMatrix<F>,
    pub bias: DVector<F>,
    activation: Act,
    regularization: Reg,
}

#[derive(Clone, Debug)]
pub struct NeuraDenseLayerPartial<F, Act, Reg, R: Rng> {
    activation: Act,
    regularization: Reg,
    output_size: usize,
    rng: R,
    phantom: PhantomData<F>,
}

impl<F: Float + std::fmt::Debug + 'static, Act: NeuraDerivable<F>, Reg: NeuraDerivable<F>>
    NeuraDenseLayer<F, Act, Reg>
{
    pub fn new(
        weights: DMatrix<F>,
        bias: DVector<F>,
        activation: Act,
        regularization: Reg,
    ) -> Self {
        assert_eq!(bias.shape().0, weights.shape().0);

        Self {
            weights,
            bias,
            activation,
            regularization,
        }
    }

    pub fn from_rng(
        input_size: usize,
        output_size: usize,
        rng: &mut impl Rng,
        activation: Act,
        regularization: Reg,
    ) -> Self
    where
        rand_distr::StandardNormal: rand_distr::Distribution<F>,
    {
        let stddev = activation.variance_hint() * 2.0 / (input_size as f64 + output_size as f64);
        let stddev = F::from(stddev).unwrap_or_else(|| {
            panic!(
                "Couldn't convert stddev ({}) to type {}",
                stddev,
                stringify!(F)
            );
        });
        let bias = F::from(activation.bias_hint()).unwrap_or_else(|| {
            panic!(
                "Couldn't convert bias ({}) to type {}",
                activation.bias_hint(),
                stringify!(F)
            );
        });

        let distribution = rand_distr::Normal::new(F::zero(), stddev)
            .expect("Couldn't create normal distribution");

        Self {
            weights: DMatrix::from_distribution(output_size, input_size, &distribution, rng),
            bias: DVector::from_element(output_size, bias),
            activation,
            regularization,
        }
    }

    pub fn new_partial<R: Rng>(
        output_size: usize,
        rng: R,
        activation: Act,
        regularization: Reg,
    ) -> NeuraDenseLayerPartial<F, Act, Reg, R> {
        NeuraDenseLayerPartial {
            activation,
            regularization,
            output_size,
            rng,
            phantom: PhantomData,
        }
    }

    pub fn input_len(&self) -> usize {
        self.weights.shape().1
    }
}

impl<F, Act, Reg, R: Rng> NeuraDenseLayerPartial<F, Act, Reg, R> {
    pub fn activation<Act2>(self, activation: Act2) -> NeuraDenseLayerPartial<F, Act2, Reg, R> {
        NeuraDenseLayerPartial {
            activation,
            regularization: self.regularization,
            output_size: self.output_size,
            rng: self.rng,
            phantom: PhantomData,
        }
    }

    pub fn regularization<Reg2>(
        self,
        regularization: Reg2,
    ) -> NeuraDenseLayerPartial<F, Act, Reg2, R> {
        NeuraDenseLayerPartial {
            activation: self.activation,
            regularization,
            output_size: self.output_size,
            rng: self.rng,
            phantom: PhantomData,
        }
    }
}

impl<F: Float, Act: NeuraDerivable<F>, Reg: NeuraDerivable<F>> NeuraShapedLayer
    for NeuraDenseLayer<F, Act, Reg>
{
    fn output_shape(&self) -> NeuraShape {
        NeuraShape::Vector(self.weights.shape().0)
    }
}

impl<
        F: Float + std::fmt::Debug + 'static,
        Act: NeuraDerivable<F>,
        Reg: NeuraDerivable<F>,
        R: Rng,
    > NeuraPartialLayer for NeuraDenseLayerPartial<F, Act, Reg, R>
where
    rand_distr::StandardNormal: rand_distr::Distribution<F>,
{
    type Constructed = NeuraDenseLayer<F, Act, Reg>;
    type Err = ();

    fn construct(self, input_shape: NeuraShape) -> Result<Self::Constructed, Self::Err> {
        let mut rng = self.rng;
        Ok(NeuraDenseLayer::from_rng(
            input_shape.size(),
            self.output_size,
            &mut rng,
            self.activation,
            self.regularization,
        ))
    }
}

impl<F: Float, Act: NeuraDerivable<F>, Reg: NeuraDerivable<F>> NeuraPartialLayer
    for NeuraDenseLayer<F, Act, Reg>
{
    type Constructed = Self;
    type Err = NeuraDimensionsMismatch;

    fn construct(self, input_shape: NeuraShape) -> Result<Self::Constructed, Self::Err> {
        if input_shape.size() != self.weights.shape().1 {
            return Err(NeuraDimensionsMismatch {
                existing: self.weights.shape().1,
                new: input_shape,
            });
        }

        Ok(self)
    }
}

impl<
        F: Float + std::fmt::Debug + 'static + std::ops::AddAssign + std::ops::MulAssign,
        Act: NeuraDerivable<F>,
        Reg: NeuraDerivable<F>,
    > NeuraLayer<DVector<F>> for NeuraDenseLayer<F, Act, Reg>
{
    type Output = DVector<F>;

    fn eval(&self, input: &DVector<F>) -> Self::Output {
        assert_eq!(input.shape().0, self.weights.shape().1);

        let evaluated = &self.weights * input + &self.bias;

        evaluated.map(|x| self.activation.eval(x))
    }
}

impl<
        F: Float + std::fmt::Debug + 'static + std::ops::AddAssign + std::ops::MulAssign,
        Act: NeuraDerivable<F>,
        Reg: NeuraDerivable<F>,
    > NeuraTrainableLayerBase for NeuraDenseLayer<F, Act, Reg>
{
    type Gradient = (DMatrix<F>, DVector<F>);

    fn default_gradient(&self) -> Self::Gradient {
        (
            DMatrix::zeros(self.weights.shape().0, self.weights.shape().1),
            DVector::zeros(self.bias.shape().0),
        )
    }

    fn apply_gradient(&mut self, gradient: &Self::Gradient) {
        self.weights += &gradient.0;
        self.bias += &gradient.1;
    }
}

impl<
        F: Float + std::fmt::Debug + 'static + std::ops::AddAssign + std::ops::MulAssign,
        Act: NeuraDerivable<F>,
        Reg: NeuraDerivable<F>,
    > NeuraTrainableLayerEval<DVector<F>> for NeuraDenseLayer<F, Act, Reg>
{
    type IntermediaryRepr = DVector<F>; // pre-activation values

    fn eval_training(&self, input: &DVector<F>) -> (Self::Output, Self::IntermediaryRepr) {
        let evaluated = &self.weights * input + &self.bias;
        let output = evaluated.map(|x| self.activation.eval(x));

        (output, evaluated)
    }
}

impl<
        F: Float + std::fmt::Debug + 'static + std::ops::AddAssign + std::ops::MulAssign,
        Act: NeuraDerivable<F>,
        Reg: NeuraDerivable<F>,
    > NeuraTrainableLayerSelf<DVector<F>> for NeuraDenseLayer<F, Act, Reg>
{
    fn regularize_layer(&self) -> Self::Gradient {
        (
            self.weights.map(|x| self.regularization.derivate(x)),
            DVector::zeros(self.bias.shape().0),
        )
    }

    fn get_gradient(
        &self,
        input: &DVector<F>,
        evaluated: &Self::IntermediaryRepr,
        epsilon: &Self::Output,
    ) -> Self::Gradient {
        // Compute delta (the input gradient of the neuron) from epsilon (the output gradient of the neuron),
        // with `self.activation'(input) ° epsilon = delta`
        let mut delta = epsilon.clone();

        for i in 0..delta.len() {
            // TODO: remove `- self.bias[i]`
            delta[i] *= self.activation.derivate(evaluated[i]);
        }

        let weights_gradient = &delta * input.transpose();

        // According to https://datascience.stackexchange.com/questions/20139/gradients-for-bias-terms-in-backpropagation
        // The gradient of the bias is equal to the delta term of the backpropagation algorithm
        let bias_gradient = delta;

        (weights_gradient, bias_gradient)
    }
}

impl<
        F: Float + std::fmt::Debug + 'static + std::ops::AddAssign + std::ops::MulAssign,
        Act: NeuraDerivable<F>,
        Reg: NeuraDerivable<F>,
    > NeuraTrainableLayerBackprop<DVector<F>> for NeuraDenseLayer<F, Act, Reg>
{
    fn backprop_layer(
        &self,
        _input: &DVector<F>,
        evaluated: &Self::IntermediaryRepr,
        epsilon: &Self::Output,
    ) -> DVector<F> {
        // Compute delta (the input gradient of the neuron) from epsilon (the output gradient of the neuron),
        // with `self.activation'(input) ° epsilon = delta`
        let mut delta = epsilon.clone();

        for i in 0..delta.len() {
            delta[i] *= self.activation.derivate(evaluated[i]);
        }

        self.weights.tr_mul(&delta)
    }
}
