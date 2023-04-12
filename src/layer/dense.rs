use super::NeuraLayer;
use crate::{derivable::NeuraDerivable, utils::{multiply_matrix_vector, reverse_dot_product, multiply_matrix_transpose_vector}, train::NeuraTrainableLayer, algebra::NeuraVectorSpace};
use rand::Rng;

#[derive(Clone, Debug)]
pub struct NeuraDenseLayer<
    Act: NeuraDerivable<f64>,
    const INPUT_LEN: usize,
    const OUTPUT_LEN: usize,
> {
    weights: [[f64; INPUT_LEN]; OUTPUT_LEN],
    bias: [f64; OUTPUT_LEN],
    activation: Act,
}

impl<Act: NeuraDerivable<f64>, const INPUT_LEN: usize, const OUTPUT_LEN: usize>
    NeuraDenseLayer<Act, INPUT_LEN, OUTPUT_LEN>
{
    pub fn new(
        weights: [[f64; INPUT_LEN]; OUTPUT_LEN],
        bias: [f64; OUTPUT_LEN],
        activation: Act,
    ) -> Self {
        Self {
            weights,
            bias,
            activation,
        }
    }

    pub fn from_rng(rng: &mut impl Rng, activation: Act) -> Self {
        let mut weights = [[0.0; INPUT_LEN]; OUTPUT_LEN];

        let multiplier = std::f64::consts::SQRT_2 / (INPUT_LEN as f64).sqrt();

        for i in 0..OUTPUT_LEN {
            for j in 0..INPUT_LEN {
                weights[i][j] = rng.gen_range(-multiplier..multiplier);
            }
        }

        Self {
            weights,
            // Biases are zero-initialized, as this shouldn't cause any issues during training
            bias: [0.0; OUTPUT_LEN],
            activation,
        }
    }
}

impl<Act: NeuraDerivable<f64>, const INPUT_LEN: usize, const OUTPUT_LEN: usize> NeuraLayer
    for NeuraDenseLayer<Act, INPUT_LEN, OUTPUT_LEN>
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

impl<Act: NeuraDerivable<f64>, const INPUT_LEN: usize, const OUTPUT_LEN: usize> NeuraTrainableLayer
    for NeuraDenseLayer<Act, INPUT_LEN, OUTPUT_LEN>
{
    type Delta = ([[f64; INPUT_LEN]; OUTPUT_LEN], [f64; OUTPUT_LEN]);

    // TODO: double-check the math in this
    fn backpropagate(&self, input: &Self::Input, epsilon: Self::Output) -> (Self::Input, Self::Delta) {
        let evaluated = multiply_matrix_vector(&self.weights, input);
        // Compute delta from epsilon, with `self.activation'(z) * epsilon = delta`
        let mut delta = epsilon.clone();
        for i in 0..OUTPUT_LEN {
            delta[i] = self.activation.derivate(evaluated[i]);
        }

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
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::derivable::activation::Relu;

    #[test]
    fn test_from_rng() {
        let mut rng = rand::thread_rng();
        let layer: NeuraDenseLayer<_, 64, 32> = NeuraDenseLayer::from_rng(&mut rng, Relu);
        let mut input = [0.0; 64];
        for x in 0..64 {
            input[x] = rng.gen();
        }
        assert!(layer.eval(&input).len() == 32);
    }
}
