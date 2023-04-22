use nalgebra::{DVector, Scalar};
use num::{traits::NumAssignOps, Float};

use super::*;

#[derive(Clone, Debug)]
pub struct NeuraSoftmaxLayer {
    shape: NeuraShape,
}

impl NeuraSoftmaxLayer {
    pub fn new() -> Self {
        Self {
            shape: NeuraShape::Vector(0),
        }
    }
}

impl<F: Float + Scalar + NumAssignOps> NeuraLayer<DVector<F>> for NeuraSoftmaxLayer {
    type Output = DVector<F>;

    fn eval(&self, input: &DVector<F>) -> Self::Output {
        let mut res = input.clone();

        let mut max = F::zero();
        for &item in &res {
            if item > max {
                max = item;
            }
        }

        let mut sum = F::zero();
        for item in &mut res {
            *item = (*item - max).exp();
            sum += *item;
        }

        res /= sum;

        res
    }
}

impl NeuraPartialLayer for NeuraSoftmaxLayer {
    type Constructed = Self;
    type Err = ();

    fn construct(self, input_shape: NeuraShape) -> Result<Self::Constructed, Self::Err> {
        Ok(Self { shape: input_shape })
    }

    fn output_shape(constructed: &Self::Constructed) -> NeuraShape {
        constructed.shape
    }
}

impl<F: Float + Scalar + NumAssignOps> NeuraTrainableLayerBase<DVector<F>> for NeuraSoftmaxLayer {
    type Gradient = ();
    type IntermediaryRepr = Self::Output; // Result of self.eval

    fn default_gradient(&self) -> Self::Gradient {
        ()
    }

    fn apply_gradient(&mut self, _gradient: &Self::Gradient) {
        // Noop
    }

    fn eval_training(&self, input: &DVector<F>) -> (Self::Output, Self::IntermediaryRepr) {
        let res = self.eval(input);
        (res.clone(), res)
    }
}

impl<F: Float + Scalar + NumAssignOps> NeuraTrainableLayerSelf<DVector<F>> for NeuraSoftmaxLayer {
    #[inline(always)]
    fn regularize_layer(&self) -> Self::Gradient {
        ()
    }

    #[inline(always)]
    fn get_gradient(
        &self,
        _input: &DVector<F>,
        _intermediary: &Self::IntermediaryRepr,
        _epsilon: &Self::Output,
    ) -> Self::Gradient {
        ()
    }
}

impl<F: Float + Scalar + NumAssignOps> NeuraTrainableLayerBackprop<DVector<F>>
    for NeuraSoftmaxLayer
{
    fn backprop_layer(
        &self,
        input: &DVector<F>,
        evaluated: &Self::IntermediaryRepr,
        epsilon: &Self::Output,
    ) -> DVector<F> {
        let mut epsilon = epsilon.clone();

        // Compute $a_{l-1,i} ° \epsilon_{l,i}$
        hadamard_product(&mut epsilon, &evaluated);

        // Compute $\sum_{k}{a_{l-1,k} \epsilon_{l,k}}$
        let sum_diagonal_terms = epsilon.sum();

        for i in 0..input.len() {
            // Multiply $\sum_{k}{a_{l-1,k} \epsilon_{l,k}}$ by $a_{l-1,i}$ and add it to $a_{l-1,i} \epsilon_{l,i}$
            epsilon[i] -= evaluated[i] * sum_diagonal_terms;
        }

        epsilon
    }
}

fn hadamard_product<F: Float + std::ops::MulAssign>(left: &mut DVector<F>, right: &DVector<F>) {
    for i in 0..left.len() {
        left[i] *= right[i];
    }
}

#[cfg(test)]
mod test {
    use nalgebra::{dvector, DMatrix};

    use crate::utils::uniform_vector;

    use super::*;

    #[test]
    fn test_softmax_eval() {
        const EPSILON: f64 = 0.000002;
        let layer = NeuraSoftmaxLayer::new();

        let result = layer.eval(&dvector![1.0, 2.0, 8.0]);

        assert!((result[0] - 0.0009088).abs() < EPSILON);
        assert!((result[1] - 0.0024704).abs() < EPSILON);
        assert!((result[2] - 0.9966208).abs() < EPSILON);
    }

    // Based on https://stats.stackexchange.com/a/306710
    #[test]
    fn test_softmax_backpropagation_two() {
        const EPSILON: f64 = 0.000001;
        let layer = NeuraSoftmaxLayer::new();

        for input1 in [0.2, 0.3, 0.5] as [f64; 3] {
            for input2 in [0.7, 1.1, 1.3] {
                let input = dvector![input1, input2];
                let sum = input1.exp() + input2.exp();
                let output = dvector![input1.exp() / sum, input2.exp() / sum];
                for epsilon1 in [1.7, 1.9, 2.3] {
                    for epsilon2 in [2.9, 3.1, 3.7] {
                        let epsilon = dvector![epsilon1, epsilon2];
                        let evaluated = layer.eval(&input);

                        let epsilon = layer.backprop_layer(&input, &evaluated, &epsilon);
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
        let layer = NeuraSoftmaxLayer::new();

        for _ in 0..100 {
            let input = uniform_vector(4);
            let evaluated = layer.eval(&input);
            let loss = uniform_vector(4);

            let mut derivative = &evaluated * evaluated.transpose();
            derivative *= -1.0;
            derivative += DMatrix::from_diagonal(&evaluated);

            let expected = derivative * &loss;
            let evaluated = layer.eval(&input);
            let actual = layer.backprop_layer(&input, &evaluated, &loss);

            for i in 0..4 {
                assert!((expected[i] - actual[i]).abs() < EPSILON);
            }
        }
    }
}
