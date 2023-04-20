use num::ToPrimitive;

use crate::{
    derivable::NeuraLoss,
    layer::NeuraTrainableLayer,
    network::{NeuraTrainableNetwork, NeuraTrainableNetworkBase},
};

pub trait NeuraOptimizerBase {
    type Output<NetworkInput, NetworkGradient>;
}

pub trait NeuraOptimizerFinal<LayerOutput>: NeuraOptimizerBase {
    fn eval_final(&self, output: LayerOutput) -> Self::Output<LayerOutput, ()>;
}

pub trait NeuraOptimizerTransient<LayerOutput>: NeuraOptimizerBase {
    fn eval_layer<
        Input,
        NetworkGradient,
        RecGradient,
        Layer: NeuraTrainableLayer<Input, Output = LayerOutput>,
    >(
        &self,
        layer: &Layer,
        input: &Input,
        rec_opt_output: Self::Output<LayerOutput, RecGradient>,
        combine_gradients: impl Fn(Layer::Gradient, RecGradient) -> NetworkGradient,
    ) -> Self::Output<Input, NetworkGradient>;
}

pub trait NeuraOptimizer<Input, Target, Trainable: NeuraTrainableNetworkBase<Input>> {
    fn get_gradient(
        &self,
        trainable: &Trainable,
        input: &Input,
        target: &Target,
    ) -> Trainable::Gradient;

    fn score(&self, trainable: &Trainable, input: &Input, target: &Target) -> f64;
}

pub struct NeuraBackprop<Loss> {
    loss: Loss,
}

impl<Loss> NeuraBackprop<Loss> {
    pub fn new(loss: Loss) -> Self {
        Self { loss }
    }
}

impl<
        Input,
        Target,
        Trainable: NeuraTrainableNetworkBase<Input>,
        Loss: NeuraLoss<Trainable::Output, Target = Target> + Clone,
    > NeuraOptimizer<Input, Target, Trainable> for NeuraBackprop<Loss>
where
    <Loss as NeuraLoss<Trainable::Output>>::Output: ToPrimitive,
    Trainable: for<'a> NeuraTrainableNetwork<Input, (&'a NeuraBackprop<Loss>, &'a Target)>,
{
    fn get_gradient(
        &self,
        trainable: &Trainable,
        input: &Input,
        target: &Target,
    ) -> Trainable::Gradient {
        let (_, gradient) = trainable.traverse(input, &(self, target));

        gradient
    }

    fn score(&self, trainable: &Trainable, input: &Input, target: &Target) -> f64 {
        let output = trainable.eval(&input);
        self.loss.eval(target, &output).to_f64().unwrap()
    }
}

impl<Loss, Target> NeuraOptimizerBase for (&NeuraBackprop<Loss>, &Target) {
    type Output<NetworkInput, NetworkGradient> = (NetworkInput, NetworkGradient); // epsilon, gradient
}

impl<LayerOutput, Target, Loss: NeuraLoss<LayerOutput, Target = Target>>
    NeuraOptimizerFinal<LayerOutput> for (&NeuraBackprop<Loss>, &Target)
{
    fn eval_final(&self, output: LayerOutput) -> Self::Output<LayerOutput, ()> {
        (self.0.loss.nabla(self.1, &output), ())
    }
}

impl<LayerOutput, Target, Loss> NeuraOptimizerTransient<LayerOutput>
    for (&NeuraBackprop<Loss>, &Target)
{
    fn eval_layer<
        Input,
        NetworkGradient,
        RecGradient,
        Layer: NeuraTrainableLayer<Input, Output = LayerOutput>,
    >(
        &self,
        layer: &Layer,
        input: &Input,
        rec_opt_output: Self::Output<LayerOutput, RecGradient>,
        combine_gradients: impl Fn(Layer::Gradient, RecGradient) -> NetworkGradient,
    ) -> Self::Output<Input, NetworkGradient> {
        let (epsilon_in, rec_gradient) = rec_opt_output;
        let (epsilon_out, layer_gradient) = layer.backprop_layer(input, epsilon_in);

        (epsilon_out, combine_gradients(layer_gradient, rec_gradient))
    }
}
