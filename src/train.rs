use crate::{algebra::NeuraVectorSpace, gradient_solver::NeuraGradientSolver, layer::*};

#[non_exhaustive]
pub struct NeuraBatchedTrainer {
    /// The learning rate of the gradient descent algorithm; the weights `W` will be updated as follows:
    /// `W += -learning_rate * gradient_average`.
    ///
    /// Defaults to `0.1`
    pub learning_rate: f64,

    /// The momentum of the gradient descent algorithm; if set to a non-zero value, then the weights `W` will be updated as follows:
    /// `W += -learning_rate * gradient_average - learning_momentum * previous_gradient`.
    /// This value should be smaller than `learning_rate`.
    ///
    /// Defaults to `0.0`
    pub learning_momentum: f64,

    /// How many gradient computations to average before updating the weights
    pub batch_size: usize,

    /// How many batches to run for; if `iterations * batch_size` exceeds the input length, then training will stop.
    /// You should use `cycle_shuffling` from the `prelude` module to avoid this.
    ///
    /// Note that this is different from epochs, which count how many times the dataset has been fully iterated over.
    pub iterations: usize,

    /// The trainer will log progress at every multiple of `log_iterations` iterations.
    /// If `log_iterations` is zero (default), then no progress will be logged.
    ///
    /// The test inputs is used to measure the score of the network.
    pub log_iterations: usize,
}

impl Default for NeuraBatchedTrainer {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            learning_momentum: 0.0,
            batch_size: 100,
            iterations: 100,
            log_iterations: 0,
        }
    }
}

impl NeuraBatchedTrainer {
    pub fn new(learning_rate: f64, iterations: usize) -> Self {
        Self {
            learning_rate,
            iterations,
            ..Default::default()
        }
    }

    pub fn with_epochs(
        learning_rate: f64,
        epochs: usize,
        batch_size: usize,
        training_size: usize,
    ) -> Self {
        Self {
            learning_rate,
            iterations: (training_size * epochs / batch_size).max(1),
            log_iterations: (training_size / batch_size).max(1),
            batch_size,
            ..Default::default()
        }
    }

    pub fn train<
        Input: Clone,
        Target: Clone,
        Network: NeuraTrainableLayerBase + NeuraTrainableLayerSelf<Input>,
        GradientSolver: NeuraGradientSolver<Input, Target, Network>,
        Inputs: IntoIterator<Item = (Input, Target)>,
    >(
        &self,
        gradient_solver: &GradientSolver,
        network: &mut Network,
        inputs: Inputs,
        test_inputs: &[(Input, Target)],
    ) -> Vec<(f64, f64)>
    where
        Network::Gradient: std::fmt::Debug,
    {
        let mut losses = Vec::new();
        let mut iter = inputs.into_iter();
        let factor = -self.learning_rate / (self.batch_size as f64);
        let momentum_factor = self.learning_momentum / self.learning_rate;
        let reg_factor = -self.learning_rate;

        // Contains `momentum_factor * factor * gradient_sum_previous_iter`
        let mut previous_gradient_sum = network.default_gradient();
        let mut train_loss = 0.0;
        'd: for iteration in 0..self.iterations {
            let mut gradient_sum = network.default_gradient();
            network.prepare_layer(true);

            for _ in 0..self.batch_size {
                if let Some((input, target)) = iter.next() {
                    let gradient = gradient_solver.get_gradient(&network, &input, &target);
                    gradient_sum.add_assign(&gradient);

                    train_loss += gradient_solver.score(&network, &input, &target);
                } else {
                    break 'd;
                }
            }

            gradient_sum.mul_assign(factor);

            // Add regularization gradient
            let mut reg_gradient = network.regularize_layer();
            reg_gradient.mul_assign(reg_factor);
            gradient_sum.add_assign(&reg_gradient);

            network.apply_gradient(&gradient_sum);

            if self.learning_momentum != 0.0 {
                network.apply_gradient(&previous_gradient_sum);
                previous_gradient_sum = gradient_sum;
                previous_gradient_sum.mul_assign(momentum_factor);
            }

            if self.log_iterations > 0 && (iteration + 1) % self.log_iterations == 0 {
                network.prepare_layer(false);
                let mut val_loss = 0.0;
                for (input, target) in test_inputs {
                    val_loss += gradient_solver.score(&network, input, target);
                }
                val_loss /= test_inputs.len() as f64;
                train_loss /= (self.batch_size * self.log_iterations) as f64;
                println!(
                    "Iteration {}, Training loss: {:.3}, Validation loss: {:.3}",
                    iteration + 1,
                    train_loss,
                    val_loss
                );

                losses.push((train_loss, val_loss));
                train_loss = 0.0;
            }
        }

        network.prepare_layer(false);

        losses
    }
}

#[cfg(test)]
mod test {
    use nalgebra::{dmatrix, dvector};

    use super::*;
    use crate::{
        assert_approx,
        derivable::{activation::Linear, loss::Euclidean, regularize::NeuraL0, NeuraLoss},
        gradient_solver::NeuraBackprop,
        layer::{dense::NeuraDenseLayer, NeuraLayer},
        network::sequential::{NeuraSequential, NeuraSequentialTail},
        neura_sequential,
    };

    #[test]
    fn test_backpropagation_simple() {
        for wa in [0.0, 0.25, 0.5, 1.0] {
            for wb in [0.0, 0.25, 0.5, 1.0] {
                let network = NeuraSequential::new(
                    NeuraDenseLayer::new(dmatrix![wa, wb], dvector![0.0], Linear, NeuraL0),
                    (),
                );

                let (gradient, _) = NeuraBackprop::new(Euclidean).get_gradient(
                    &network,
                    &dvector![1.0, 1.0],
                    &dvector![0.0],
                );

                let expected = wa + wb;
                assert!((gradient.0[(0, 0)] - expected) < 0.001);
                assert!((gradient.0[(0, 1)] - expected) < 0.001);
            }
        }
    }

    #[test]
    fn test_backpropagation_complex() {
        const EPSILON: f64 = 0.00001;
        // Test that we get the same values as https://hmkcode.com/ai/backpropagation-step-by-step/
        let network = neura_sequential![
            NeuraDenseLayer::new(
                dmatrix![0.11, 0.21; 0.12, 0.08],
                dvector![0.0, 0.0],
                Linear,
                NeuraL0
            ),
            NeuraDenseLayer::new(dmatrix![0.14, 0.15], dvector![0.0], Linear, NeuraL0)
        ];

        let input = dvector![2.0, 3.0];
        let target = dvector![1.0];

        let intermediary = network.clone().trim_tail().eval(&input);
        assert_approx!(0.85, intermediary[0], EPSILON);
        assert_approx!(0.48, intermediary[1], EPSILON);
        assert_approx!(0.191, network.eval(&input)[0], EPSILON);

        assert_approx!(0.327, Euclidean.eval(&target, &network.eval(&input)), 0.001);

        let delta = network.eval(&input)[0] - target[0];

        let (gradient_first, gradient_second) =
            NeuraBackprop::new(Euclidean).get_gradient(&network, &input, &target);
        let gradient_first = gradient_first.0;
        let gradient_second = gradient_second.0 .0;

        assert_approx!(gradient_second[0], intermediary[0] * delta, EPSILON);
        assert_approx!(gradient_second[1], intermediary[1] * delta, EPSILON);

        assert_approx!(gradient_first[(0, 0)], input[0] * delta * 0.14, EPSILON);
        assert_approx!(gradient_first[(0, 1)], input[1] * delta * 0.14, EPSILON);

        assert_approx!(gradient_first[(1, 0)], input[0] * delta * 0.15, EPSILON);
        assert_approx!(gradient_first[(1, 1)], input[1] * delta * 0.15, EPSILON);
    }
}
