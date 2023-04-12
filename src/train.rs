use crate::{
    // utils::{assign_add_vector, chunked},
    algebra::NeuraVectorSpace,
    derivable::NeuraLoss,
    layer::NeuraLayer,
    network::NeuraNetwork,
};

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
}

pub trait NeuraTrainer<Output, Target = Output> {
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

impl<const N: usize, Loss: NeuraLoss<Input = [f64; N]> + Clone> NeuraTrainer<[f64; N], Loss::Target>
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

pub fn train_batched<
    Output,
    Target,
    Trainer: NeuraTrainer<Output, Target>,
    Layer: NeuraLayer,
    ChildNetwork,
    Inputs: IntoIterator<Item = (Layer::Input, Target)>,
>(
    network: &mut NeuraNetwork<Layer, ChildNetwork>,
    inputs: Inputs,
    test_inputs: &[(Layer::Input, Target)],
    trainer: Trainer,
    learning_rate: f64,
    batch_size: usize,
    epochs: usize,
) where
    NeuraNetwork<Layer, ChildNetwork>: NeuraTrainable<Input = Layer::Input, Output = Output>,
    Inputs::IntoIter: Clone,
{
    // TODO: apply shuffling?
    let mut iter = inputs.into_iter().cycle();
    let factor = -learning_rate / (batch_size as f64);

    'd: for epoch in 0..epochs {
        let mut gradient_sum = <NeuraNetwork<Layer, ChildNetwork> as NeuraTrainable>::Delta::zero();

        for _ in 0..batch_size {
            if let Some((input, target)) = iter.next() {
                let gradient = trainer.get_gradient(&network, &input, &target);
                gradient_sum.add_assign(&gradient);
            } else {
                break 'd;
            }
        }

        gradient_sum.mul_assign(factor);
        network.apply_gradient(&gradient_sum);

        let mut loss_sum = 0.0;
        for (input, target) in test_inputs {
            loss_sum += trainer.score(&network, input, target);
        }
        loss_sum /= test_inputs.len() as f64;
        println!("Epoch {epoch}, Loss: {:.3}", loss_sum);
    }
}
