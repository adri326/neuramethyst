use crate::{
    algebra::{NeuraVector, NeuraVectorSpace},
    derivable::NeuraLoss,
    layer::NeuraLayer,
    network::{sequential::NeuraSequential, NeuraTrainableNetwork},
};

pub trait NeuraGradientSolver<Output, Target = Output> {
    fn get_gradient<Layer: NeuraLayer, ChildNetwork>(
        &self,
        trainable: &NeuraSequential<Layer, ChildNetwork>,
        input: &Layer::Input,
        target: &Target,
    ) -> <NeuraSequential<Layer, ChildNetwork> as NeuraTrainableNetwork>::Delta
    where
        NeuraSequential<Layer, ChildNetwork>:
            NeuraTrainableNetwork<Input = Layer::Input, Output = Output>;

    fn score<Layer: NeuraLayer, ChildNetwork>(
        &self,
        trainable: &NeuraSequential<Layer, ChildNetwork>,
        input: &Layer::Input,
        target: &Target,
    ) -> f64
    where
        NeuraSequential<Layer, ChildNetwork>:
            NeuraTrainableNetwork<Input = Layer::Input, Output = Output>;
}

#[non_exhaustive]
pub struct NeuraBackprop<Loss: NeuraLoss + Clone> {
    loss: Loss,
}

impl<Loss: NeuraLoss + Clone> NeuraBackprop<Loss> {
    pub fn new(loss: Loss) -> Self {
        Self { loss }
    }
}

impl<const N: usize, Loss: NeuraLoss<Input = NeuraVector<N, f64>> + Clone>
    NeuraGradientSolver<NeuraVector<N, f64>, Loss::Target> for NeuraBackprop<Loss>
{
    fn get_gradient<Layer: NeuraLayer, ChildNetwork>(
        &self,
        trainable: &NeuraSequential<Layer, ChildNetwork>,
        input: &Layer::Input,
        target: &Loss::Target,
    ) -> <NeuraSequential<Layer, ChildNetwork> as NeuraTrainableNetwork>::Delta
    where
        NeuraSequential<Layer, ChildNetwork>:
            NeuraTrainableNetwork<Input = Layer::Input, Output = NeuraVector<N, f64>>,
    {
        trainable.backpropagate(input, target, self.loss.clone()).1
    }

    fn score<Layer: NeuraLayer, ChildNetwork>(
        &self,
        trainable: &NeuraSequential<Layer, ChildNetwork>,
        input: &Layer::Input,
        target: &Loss::Target,
    ) -> f64
    where
        NeuraSequential<Layer, ChildNetwork>:
            NeuraTrainableNetwork<Input = Layer::Input, Output = NeuraVector<N, f64>>,
    {
        let output = trainable.eval(&input);
        self.loss.eval(target, &output)
    }
}

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

    pub fn train<
        Output,
        Target: Clone,
        GradientSolver: NeuraGradientSolver<Output, Target>,
        Layer: NeuraLayer,
        ChildNetwork,
        Inputs: IntoIterator<Item = (Layer::Input, Target)>,
    >(
        &self,
        gradient_solver: GradientSolver,
        network: &mut NeuraSequential<Layer, ChildNetwork>,
        inputs: Inputs,
        test_inputs: &[(Layer::Input, Target)],
    ) where
        NeuraSequential<Layer, ChildNetwork>:
            NeuraTrainableNetwork<Input = Layer::Input, Output = Output>,
        Layer::Input: Clone,
    {
        let mut iter = inputs.into_iter();
        let factor = -self.learning_rate / (self.batch_size as f64);
        let momentum_factor = self.learning_momentum / self.learning_rate;
        let reg_factor = -self.learning_rate;

        // Contains `momentum_factor * factor * gradient_sum_previous_iter`
        let mut previous_gradient_sum =
            Box::<<NeuraSequential<Layer, ChildNetwork> as NeuraTrainableNetwork>::Delta>::zero();
        'd: for iteration in 0..self.iterations {
            let mut gradient_sum = Box::<
                <NeuraSequential<Layer, ChildNetwork> as NeuraTrainableNetwork>::Delta,
            >::zero();
            network.prepare_epoch();

            for _ in 0..self.batch_size {
                if let Some((input, target)) = iter.next() {
                    let gradient =
                        Box::new(gradient_solver.get_gradient(&network, &input, &target));
                    gradient_sum.add_assign(&gradient);
                } else {
                    break 'd;
                }
            }

            gradient_sum.mul_assign(factor);

            // Add regularization gradient
            let mut reg_gradient = Box::new(network.regularize());
            reg_gradient.mul_assign(reg_factor);
            gradient_sum.add_assign(&reg_gradient);

            network.apply_gradient(&gradient_sum);

            if self.learning_momentum != 0.0 {
                network.apply_gradient(&previous_gradient_sum);
                previous_gradient_sum = gradient_sum;
                previous_gradient_sum.mul_assign(momentum_factor);
            }

            if self.log_iterations > 0 && (iteration + 1) % self.log_iterations == 0 {
                network.cleanup();
                let mut loss_sum = 0.0;
                for (input, target) in test_inputs {
                    loss_sum += gradient_solver.score(&network, input, target);
                }
                loss_sum /= test_inputs.len() as f64;
                println!("Iteration {}, Loss: {:.3}", iteration + 1, loss_sum);
            }
        }

        network.cleanup();
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        assert_approx,
        derivable::{activation::Linear, loss::Euclidean, regularize::NeuraL0},
        layer::NeuraDenseLayer,
        network::sequential::NeuraSequentialTail,
        neura_sequential,
    };

    #[test]
    fn test_backpropagation_simple() {
        for wa in [0.0, 0.25, 0.5, 1.0] {
            for wb in [0.0, 0.25, 0.5, 1.0] {
                let network = NeuraSequential::new(
                    NeuraDenseLayer::new([[wa, wb]].into(), [0.0].into(), Linear, NeuraL0),
                    (),
                );

                let gradient = NeuraBackprop::new(Euclidean).get_gradient(
                    &network,
                    &[1.0, 1.0].into(),
                    &[0.0].into(),
                );

                let expected = wa + wb;
                assert!((gradient.0[0][0] - expected) < 0.001);
                assert!((gradient.0[0][1] - expected) < 0.001);
            }
        }
    }

    #[test]
    fn test_backpropagation_complex() {
        const EPSILON: f64 = 0.00001;
        // Test that we get the same values as https://hmkcode.com/ai/backpropagation-step-by-step/
        let network = neura_sequential![
            NeuraDenseLayer::new(
                [[0.11, 0.21], [0.12, 0.08]].into(),
                [0.0; 2].into(),
                Linear,
                NeuraL0
            ),
            NeuraDenseLayer::new([[0.14, 0.15]].into(), [0.0].into(), Linear, NeuraL0)
        ];

        let input = [2.0, 3.0];
        let target = [1.0];

        let intermediary = network.clone().trim_tail().eval(&input.into());
        assert_approx!(0.85, intermediary[0], EPSILON);
        assert_approx!(0.48, intermediary[1], EPSILON);
        assert_approx!(0.191, network.eval(&input.into())[0], EPSILON);

        assert_approx!(
            0.327,
            Euclidean.eval(&target.into(), &network.eval(&input.into())),
            0.001
        );

        let delta = network.eval(&input.into())[0] - target[0];

        let (gradient_first, gradient_second) =
            NeuraBackprop::new(Euclidean).get_gradient(&network, &input.into(), &target.into());
        let gradient_first = gradient_first.0;
        let gradient_second = gradient_second.0[0];

        assert_approx!(gradient_second[0], intermediary[0] * delta, EPSILON);
        assert_approx!(gradient_second[1], intermediary[1] * delta, EPSILON);

        assert_approx!(gradient_first[0][0], input[0] * delta * 0.14, EPSILON);
        assert_approx!(gradient_first[0][1], input[1] * delta * 0.14, EPSILON);

        assert_approx!(gradient_first[1][0], input[0] * delta * 0.15, EPSILON);
        assert_approx!(gradient_first[1][1], input[1] * delta * 0.15, EPSILON);
    }
}
