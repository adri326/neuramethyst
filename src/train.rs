use crate::{
    algebra::NeuraVectorSpace,
    derivable::NeuraLoss,
    layer::NeuraLayer,
    network::NeuraNetwork,
};

// TODO: move this to layer/mod.rs
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

    /// Applies `δW_l` to the weights of the layer
    fn apply_gradient(&mut self, gradient: &Self::Delta);

    /// Called before an epoch begins, to allow the layer to set itself up for training.
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

    /// Called before an epoch begins, to allow the network to set itself up for training.
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

impl<const N: usize, Loss: NeuraLoss<Input = [f64; N]> + Clone> NeuraGradientSolver<[f64; N], Loss::Target>
    for NeuraBackprop<Loss>
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
        self.loss.eval(target, &trainable.eval(&input))
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

    /// How many batches to run for; if `epochs * batch_size` exceeds the input length, then training will stop.
    /// You should use `cycle_shuffling` from the `prelude` module to avoid this.
    pub epochs: usize,

    /// The trainer will log progress at every multiple of `log_epochs` steps.
    /// If `log_epochs` is zero (default), then no progress will be logged.
    ///
    /// The test inputs is used to measure the score of the network.
    pub log_epochs: usize,
}

impl Default for NeuraBatchedTrainer {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            learning_momentum: 0.0,
            batch_size: 100,
            epochs: 100,
            log_epochs: 0,
        }
    }
}

impl NeuraBatchedTrainer {
    pub fn new(learning_rate: f64, epochs: usize) -> Self {
        Self {
            learning_rate,
            epochs,
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
        // TODO: apply shuffling?
        let mut iter = inputs.into_iter();
        let factor = -self.learning_rate / (self.batch_size as f64);
        let momentum_factor = self.learning_momentum / self.learning_rate;

        // Contains `momentum_factor * factor * gradient_sum_previous_iter`
        let mut previous_gradient_sum = <NeuraNetwork<Layer, ChildNetwork> as NeuraTrainable>::Delta::zero();
        'd: for epoch in 0..self.epochs {
            let mut gradient_sum = <NeuraNetwork<Layer, ChildNetwork> as NeuraTrainable>::Delta::zero();
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
            network.apply_gradient(&gradient_sum);

            if self.learning_momentum != 0.0 {
                network.apply_gradient(&previous_gradient_sum);
                previous_gradient_sum = gradient_sum;
                previous_gradient_sum.mul_assign(momentum_factor);
            }

            if self.log_epochs > 0 && (epoch + 1) % self.log_epochs == 0 {
                network.cleanup();
                let mut loss_sum = 0.0;
                for (input, target) in test_inputs {
                    loss_sum += gradient_solver.score(&network, input, target);
                }
                loss_sum /= test_inputs.len() as f64;
                println!("Epoch {}, Loss: {:.3}", epoch + 1, loss_sum);
            }
        }

        network.cleanup();
    }
}

#[cfg(test)]
mod test {
    use crate::{layer::NeuraDenseLayer, derivable::{activation::Linear, loss::Euclidean}};
    use super::*;

    #[test]
    fn test_backpropagation_simple() {
        for wa in [0.0, 0.25, 0.5, 1.0] {
            for wb in [0.0, 0.25, 0.5, 1.0] {
                let network = NeuraNetwork::new(
                    NeuraDenseLayer::new([[wa, wb]], [0.0], Linear),
                    ()
                );

                let gradient = NeuraBackprop::new(Euclidean).get_gradient(
                    &network,
                    &[1.0, 1.0],
                    &[0.0]
                );

                let expected = wa + wb;
                assert!((gradient.0[0][0] - expected) < 0.001);
                assert!((gradient.0[0][1] - expected) < 0.001);
            }
        }
    }
}
