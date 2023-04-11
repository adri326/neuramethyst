use super::NeuraLayer;
use crate::{derivable::NeuraDerivable, utils::multiply_matrix_vector};
use rand::Rng;

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
                weights[i][j] = rng.gen::<f64>() * multiplier;
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
