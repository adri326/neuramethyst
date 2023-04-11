use crate::{
    derivable::NeuraLoss,
    layer::NeuraLayer,
    network::NeuraNetwork,
    // utils::{assign_add_vector, chunked},
    algebra::NeuraAddAssign,
};


pub trait NeuraTrainableLayer: NeuraLayer {
    type Delta: NeuraAddAssign;

    /// Computes the backpropagation term and the derivative of the internal weights,
    /// using the `input` vector outputted by the previous layer and the backpropagation term `epsilon` of the next layer.
    ///
    /// Note: we introduce the term `epsilon`, which together with the activation of the current function can be used to compute `delta_l`:
    /// ```no_rust
    /// f_l'(a_l) * epsilon_l = delta_l
    /// ```
    ///
    /// The function should then return a pair `(epsilon_{l-1}, δW_l)`,
    /// with `epsilon_{l-1}` being multiplied by `f_{l-1}'(activation)`.
    fn backpropagate(&self, input: &Self::Input, epsilon: Self::Output) -> (Self::Input, Self::Delta);
}

pub trait NeuraTrainable: NeuraLayer {
    type Delta: NeuraAddAssign;

    fn backpropagate<Loss: NeuraLoss<Self::Output>>(&self, input: &Self::Input, target: Loss::Target, loss: Loss) -> (Self::Input, Self::Delta);
}

pub trait NeuraTrainer<F, Loss: NeuraLoss<F>> {
    fn get_gradient<Layer: NeuraLayer, ChildNetwork>(
        &self,
        trainable: &NeuraNetwork<Layer, ChildNetwork>,
        input: &Layer::Input,
        target: Loss::Target,
        loss: Loss,
    ) -> <NeuraNetwork<Layer, ChildNetwork> as NeuraTrainable>::Delta where
        NeuraNetwork<Layer, ChildNetwork>: NeuraTrainable<Input = Layer::Input, Output = F>
    ;
}

#[non_exhaustive]
pub struct NeuraBackprop {
    pub epsilon: f64,
    pub batch_size: usize,
}

impl<const N: usize, Loss: NeuraLoss<[f64; N]>> NeuraTrainer<[f64; N], Loss> for NeuraBackprop {
    fn get_gradient<Layer: NeuraLayer, ChildNetwork>(
        &self,
        trainable: &NeuraNetwork<Layer, ChildNetwork>,
        input: &Layer::Input,
        target: Loss::Target,
        loss: Loss,
    ) -> <NeuraNetwork<Layer, ChildNetwork> as NeuraTrainable>::Delta where
        NeuraNetwork<Layer, ChildNetwork>: NeuraTrainable<Input = Layer::Input, Output = [f64; N]>,
    {
        trainable.backpropagate(input, target, loss).1
    }
}
