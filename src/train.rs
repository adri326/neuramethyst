use crate::{
    algebra::NeuraVectorSpace, derivable::NeuraLoss, layer::NeuraLayer, network::NeuraNetwork,
};

// TODO: move this trait to layer/mod.rs
pub trait NeuraTrainableLayer: NeuraLayer {
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

pub trait NeuraTrainable: NeuraLayer {
    type Delta: NeuraVectorSpace;

    fn apply_gradient(&mut self, gradient: &Self::Delta);

    /// Should implement the backpropagation algorithm, see `NeuraTrainableLayer::backpropagate` for more information.
    fn backpropagate<Loss: NeuraLoss<Input = Self::Output>>(
        &self,
        input: &Self::Input,
        target: &Loss::Target,
        loss: Loss,
    ) -> (Self::Input, Self::Delta);

    /// Should return the regularization gradient
    fn regularize(&self) -> Self::Delta;

    /// Called before an iteration begins, to allow the network to set itself up for training.
    fn prepare_epoch(&mut self);

    /// Called at the end of training, to allow the network to clean itself up
    fn cleanup(&mut self);
}

pub trait NeuraGradientSolver<Output, Target = Output> {
    fn get_gradient<Layer: NeuraLayer, ChildNetwork>(
        &self,
        trainable: &NeuraNetwork<Layer, ChildNetwork>,
        input: &Layer::Input,
        target: &Target,
    ) -> <NeuraNetwork<Layer, ChildNetwork> as NeuraTrainable>::Delta
    where
        NeuraNetwork<Layer, ChildNetwork>: NeuraTrainable<Input = Layer::Input, Output = Output>;

    fn score<Layer: NeuraLayer, ChildNetwork>(
        &self,
        trainable: &NeuraNetwork<Layer, ChildNetwork>,
        input: &Layer::Input,
        target: &Target,
    ) -> f64
    where
        NeuraNetwork<Layer, ChildNetwork>: NeuraTrainable<Input = Layer::Input, Output = Output>;
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

impl<const N: usize, Loss: NeuraLoss<Input = [f64; N]> + Clone>
    NeuraGradientSolver<[f64; N], Loss::Target> for NeuraBackprop<Loss>
{
    fn get_gradient<Layer: NeuraLayer, ChildNetwork>(
        &self,
        trainable: &NeuraNetwork<Layer, ChildNetwork>,
        input: &Layer::Input,
        target: &Loss::Target,
    ) -> <NeuraNetwork<Layer, ChildNetwork> as NeuraTrainable>::Delta
    where
        NeuraNetwork<Layer, ChildNetwork>: NeuraTrainable<Input = Layer::Input, Output = [f64; N]>,
    {
        trainable.backpropagate(input, target, self.loss.clone()).1
    }

    fn score<Layer: NeuraLayer, ChildNetwork>(
        &self,
        trainable: &NeuraNetwork<Layer, ChildNetwork>,
        input: &Layer::Input,
        target: &Loss::Target,
    ) -> f64
    where
        NeuraNetwork<Layer, ChildNetwork>: NeuraTrainable<Input = Layer::Input, Output = [f64; N]>,
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
        network: &mut NeuraNetwork<Layer, ChildNetwork>,
        inputs: Inputs,
        test_inputs: &[(Layer::Input, Target)],
    ) where
        NeuraNetwork<Layer, ChildNetwork>: NeuraTrainable<Input = Layer::Input, Output = Output>,
        Layer::Input: Clone,
    {
        let mut iter = inputs.into_iter();
        let factor = -self.learning_rate / (self.batch_size as f64);
        let momentum_factor = self.learning_momentum / self.learning_rate;
        let reg_factor = -self.learning_rate;

        // Contains `momentum_factor * factor * gradient_sum_previous_iter`
        let mut previous_gradient_sum =
            <NeuraNetwork<Layer, ChildNetwork> as NeuraTrainable>::Delta::zero();
        'd: for iteration in 0..self.iterations {
            let mut gradient_sum =
                <NeuraNetwork<Layer, ChildNetwork> as NeuraTrainable>::Delta::zero();
            network.prepare_epoch();

            for _ in 0..self.batch_size {
                if let Some((input, target)) = iter.next() {
                    let gradient = gradient_solver.get_gradient(&network, &input, &target);
                    gradient_sum.add_assign(&gradient);
                } else {
                    break 'd;
                }
            }

            gradient_sum.mul_assign(factor);

            // Add regularization gradient
            let mut reg_gradient = network.regularize();
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
        network::NeuraNetworkTail,
        neura_network,
    };

    #[test]
    fn test_backpropagation_simple() {
        for wa in [0.0, 0.25, 0.5, 1.0] {
            for wb in [0.0, 0.25, 0.5, 1.0] {
                let network =
                    NeuraNetwork::new(NeuraDenseLayer::new([[wa, wb]], [0.0], Linear, NeuraL0), ());

                let gradient =
                    NeuraBackprop::new(Euclidean).get_gradient(&network, &[1.0, 1.0], &[0.0]);

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
        let network = neura_network![
            NeuraDenseLayer::new([[0.11, 0.21], [0.12, 0.08]], [0.0; 2], Linear, NeuraL0),
            NeuraDenseLayer::new([[0.14, 0.15]], [0.0], Linear, NeuraL0)
        ];

        let input = [2.0, 3.0];
        let target = [1.0];

        let intermediary = network.clone().trim_tail().eval(&input);
        assert_approx!(0.85, intermediary[0], EPSILON);
        assert_approx!(0.48, intermediary[1], EPSILON);
        assert_approx!(0.191, network.eval(&input)[0], EPSILON);

        assert_approx!(0.327, Euclidean.eval(&target, &network.eval(&input)), 0.001);

        let delta = network.eval(&input)[0] - target[0];

        let (gradient_first, gradient_second) =
            NeuraBackprop::new(Euclidean).get_gradient(&network, &input, &target);
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
