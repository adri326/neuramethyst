use super::NeuraLayer;
use crate::{
    algebra::NeuraVectorSpace,
    derivable::NeuraDerivable,
    train::NeuraTrainableLayer,
    utils::{multiply_matrix_transpose_vector, multiply_matrix_vector, reverse_dot_product},
};

use rand::Rng;
use rand_distr::Distribution;

#[derive(Clone, Debug)]
pub struct NeuraDenseLayer<
    Act: NeuraDerivable<f64>,
    Reg: NeuraDerivable<f64>,
    const INPUT_LEN: usize,
    const OUTPUT_LEN: usize,
> {
    weights: [[f64; INPUT_LEN]; OUTPUT_LEN],
    bias: [f64; OUTPUT_LEN],
    activation: Act,
    regularization: Reg,
}

impl<
        Act: NeuraDerivable<f64>,
        Reg: NeuraDerivable<f64>,
        const INPUT_LEN: usize,
        const OUTPUT_LEN: usize,
    > NeuraDenseLayer<Act, Reg, INPUT_LEN, OUTPUT_LEN>
{
    pub fn new(
        weights: [[f64; INPUT_LEN]; OUTPUT_LEN],
        bias: [f64; OUTPUT_LEN],
        activation: Act,
        regularization: Reg,
    ) -> Self {
        Self {
            weights,
            bias,
            activation,
            regularization,
        }
    }

    pub fn from_rng(rng: &mut impl Rng, activation: Act, regularization: Reg) -> Self {
        let mut weights = [[0.0; INPUT_LEN]; OUTPUT_LEN];

        // Use Xavier (or He) initialisation, using the harmonic mean
        // Ref: https://www.deeplearning.ai/ai-notes/initialization/index.html
        let distribution = rand_distr::Normal::new(
            0.0,
            activation.variance_hint() * 2.0 / (INPUT_LEN as f64 + OUTPUT_LEN as f64),
        )
        .unwrap();
        // let distribution = rand_distr::Uniform::new(-0.5, 0.5);

        for i in 0..OUTPUT_LEN {
            for j in 0..INPUT_LEN {
                weights[i][j] = distribution.sample(rng);
            }
        }

        Self {
            weights,
            // Biases are initialized based on the activation's hint
            bias: [activation.bias_hint(); OUTPUT_LEN],
            activation,
            regularization,
        }
    }
}

impl<
        Act: NeuraDerivable<f64>,
        Reg: NeuraDerivable<f64>,
        const INPUT_LEN: usize,
        const OUTPUT_LEN: usize,
    > NeuraLayer for NeuraDenseLayer<Act, Reg, INPUT_LEN, OUTPUT_LEN>
{
    type Input = [f64; INPUT_LEN];

    type Output = [f64; OUTPUT_LEN];

    fn eval(&self, input: &Self::Input) -> Self::Output {
        let mut result = multiply_matrix_vector(&self.weights, input);

        for i in 0..OUTPUT_LEN {
            result[i] = self.activation.eval(result[i] + self.bias[i]);
        }

        result
    }
}

impl<
        Act: NeuraDerivable<f64>,
        Reg: NeuraDerivable<f64>,
        const INPUT_LEN: usize,
        const OUTPUT_LEN: usize,
    > NeuraTrainableLayer for NeuraDenseLayer<Act, Reg, INPUT_LEN, OUTPUT_LEN>
{
    type Delta = ([[f64; INPUT_LEN]; OUTPUT_LEN], [f64; OUTPUT_LEN]);

    fn backpropagate(
        &self,
        input: &Self::Input,
        epsilon: Self::Output,
    ) -> (Self::Input, Self::Delta) {
        let evaluated = multiply_matrix_vector(&self.weights, input);
        // Compute delta (the input gradient of the neuron) from epsilon (the output gradient of the neuron),
        // with `self.activation'(input) ° epsilon = delta`
        let mut delta = epsilon.clone();
        for i in 0..OUTPUT_LEN {
            delta[i] *= self.activation.derivate(evaluated[i]);
        }

        // Compute the weight gradient
        let weights_gradient = reverse_dot_product(&delta, input);

        // According to https://datascience.stackexchange.com/questions/20139/gradients-for-bias-terms-in-backpropagation
        // The gradient of the bias is equal to the delta term of the backpropagation algorithm
        let bias_gradient = delta;

        let new_epsilon = multiply_matrix_transpose_vector(&self.weights, &delta);

        (new_epsilon, (weights_gradient, bias_gradient))
    }

    fn apply_gradient(&mut self, gradient: &Self::Delta) {
        NeuraVectorSpace::add_assign(&mut self.weights, &gradient.0);
        NeuraVectorSpace::add_assign(&mut self.bias, &gradient.1);
    }

    fn regularize(&self) -> Self::Delta {
        let mut res = ([[0.0; INPUT_LEN]; OUTPUT_LEN], [0.0; OUTPUT_LEN]);

        for i in 0..OUTPUT_LEN {
            for j in 0..INPUT_LEN {
                res.0[i][j] = self.regularization.derivate(self.weights[i][j]);
            }
        }

        // Note: biases aren't taken into account here, as per https://stats.stackexchange.com/questions/153605/no-regularisation-term-for-bias-unit-in-neural-network

        res
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::derivable::{activation::Relu, regularize::NeuraL0};

    #[test]
    fn test_from_rng() {
        let mut rng = rand::thread_rng();
        let layer: NeuraDenseLayer<_, _, 64, 32> =
            NeuraDenseLayer::from_rng(&mut rng, Relu, NeuraL0);
        let mut input = [0.0; 64];
        for x in 0..64 {
            input[x] = rng.gen();
        }
        assert!(layer.eval(&input).len() == 32);
    }
}
