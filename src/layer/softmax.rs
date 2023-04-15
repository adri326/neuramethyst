use crate::{train::NeuraTrainableLayer, utils::multiply_vectors_pointwise};

use super::NeuraLayer;

#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct NeuraSoftmaxLayer<const LENGTH: usize>;

impl<const LENGTH: usize> NeuraSoftmaxLayer<LENGTH> {
    pub fn new() -> Self {
        Self
    }
}

impl<const LENGTH: usize> NeuraLayer for NeuraSoftmaxLayer<LENGTH> {
    type Input = [f64; LENGTH];
    type Output = [f64; LENGTH];

    fn eval(&self, input: &Self::Input) -> Self::Output {
        let mut res = input.clone();

        let mut max = 0.0;
        for item in &res {
            if *item > max {
                max = *item;
            }
        }

        for item in &mut res {
            *item = (*item - max).exp();
        }

        let mut sum = 0.0;
        for item in &res {
            sum += item;
        }

        for item in &mut res {
            *item /= sum;
        }

        res
    }
}

impl<const LENGTH: usize> NeuraTrainableLayer for NeuraSoftmaxLayer<LENGTH> {
    type Delta = ();

    fn backpropagate(
        &self,
        input: &Self::Input,
        mut epsilon: Self::Output,
    ) -> (Self::Input, Self::Delta) {
        // Note: a constant value can be added to `input` to bring it to increase precision
        let evaluated = self.eval(input);

        // Compute $a_{l-1,i} \epsilon_{l,i}$
        epsilon = multiply_vectors_pointwise(&epsilon, &evaluated);

        // Compute $\sum_{k}{a_{l-1,k} \epsilon_{l,k}}$
        let sum_diagonal_terms: f64 = epsilon.iter().copied().sum();

        for i in 0..LENGTH {
            // Multiply $\sum_{k}{a_{l-1,k} \epsilon_{l,k}}$ by $a_{l-1,i}$ and add it to $a_{l-1,i} \epsilon_{l,i}$
            epsilon[i] -= evaluated[i] * sum_diagonal_terms;
        }

        (epsilon, ())
    }

    fn regularize(&self) -> Self::Delta {
        ()
    }

    fn apply_gradient(&mut self, _gradient: &Self::Delta) {
        // Noop
    }
}

#[cfg(test)]
mod test {
    use crate::algebra::NeuraVectorSpace;
    use crate::utils::{
        matrix_from_diagonal, multiply_matrix_vector, reverse_dot_product, uniform_vector,
    };

    use super::*;

    #[test]
    fn test_softmax_eval() {
        const EPSILON: f64 = 0.000002;
        let layer = NeuraSoftmaxLayer::new() as NeuraSoftmaxLayer<3>;

        let result = layer.eval(&[1.0, 2.0, 8.0]);

        assert!((result[0] - 0.0009088).abs() < EPSILON);
        assert!((result[1] - 0.0024704).abs() < EPSILON);
        assert!((result[2] - 0.9966208).abs() < EPSILON);
    }

    // Based on https://stats.stackexchange.com/a/306710
    #[test]
    fn test_softmax_backpropagation_two() {
        const EPSILON: f64 = 0.000001;
        let layer = NeuraSoftmaxLayer::new() as NeuraSoftmaxLayer<2>;

        for input1 in [0.2, 0.3, 0.5] as [f64; 3] {
            for input2 in [0.7, 1.1, 1.3] {
                let input = [input1, input2];
                let sum = input1.exp() + input2.exp();
                let output = [input1.exp() / sum, input2.exp() / sum];
                for epsilon1 in [1.7, 1.9, 2.3] {
                    for epsilon2 in [2.9, 3.1, 3.7] {
                        let epsilon = [epsilon1, epsilon2];

                        let (epsilon, _) = layer.backpropagate(&input, epsilon);
                        let expected = [
                            output[0] * (1.0 - output[0]) * epsilon1
                                - output[1] * output[0] * epsilon2,
                            output[1] * (1.0 - output[1]) * epsilon2
                                - output[1] * output[0] * epsilon1,
                        ];

                        assert!((epsilon[0] - expected[0]).abs() < EPSILON);
                        assert!((epsilon[1] - expected[1]).abs() < EPSILON);
                    }
                }
            }
        }
    }

    // Based on https://e2eml.school/softmax.html
    #[test]
    fn test_softmax_backpropagation() {
        const EPSILON: f64 = 0.000001;
        let layer = NeuraSoftmaxLayer::new() as NeuraSoftmaxLayer<4>;

        for _ in 0..100 {
            let input: [f64; 4] = uniform_vector();
            let evaluated = layer.eval(&input);
            let loss: [f64; 4] = uniform_vector();

            let mut derivative = reverse_dot_product(&evaluated, &evaluated);
            derivative.mul_assign(-1.0);
            derivative.add_assign(&matrix_from_diagonal(&evaluated));

            let expected = multiply_matrix_vector(&derivative, &loss);
            let (actual, _) = layer.backpropagate(&input, loss);

            for i in 0..4 {
                assert!((expected[i] - actual[i]).abs() < EPSILON);
            }
        }
    }
}
