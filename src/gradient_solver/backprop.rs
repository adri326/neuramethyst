use num::ToPrimitive;

use crate::{
    derivable::NeuraLoss, layer::NeuraTrainableLayerBackprop, layer::NeuraTrainableLayerSelf,
    network::NeuraTrainableNetworkBase,
};

use super::*;

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
    > NeuraGradientSolver<Input, Target, Trainable> for NeuraBackprop<Loss>
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

impl<Loss, Target> NeuraGradientSolverBase for (&NeuraBackprop<Loss>, &Target) {
    type Output<NetworkInput, NetworkGradient> = (NetworkInput, NetworkGradient); // epsilon, gradient
}

impl<LayerOutput, Target, Loss: NeuraLoss<LayerOutput, Target = Target>>
    NeuraGradientSolverFinal<LayerOutput> for (&NeuraBackprop<Loss>, &Target)
{
    fn eval_final(&self, output: LayerOutput) -> Self::Output<LayerOutput, ()> {
        (self.0.loss.nabla(self.1, &output), ())
    }
}

impl<
        Input,
        Target,
        Loss,
        Layer: NeuraTrainableLayerBackprop<Input> + NeuraTrainableLayerSelf<Input>,
    > NeuraGradientSolverTransient<Input, Layer> for (&NeuraBackprop<Loss>, &Target)
{
    fn eval_layer<NetworkGradient, RecGradient>(
        &self,
        layer: &Layer,
        input: &Input,
        _output: &Layer::Output,
        intermediary: &Layer::IntermediaryRepr,
        rec_opt_output: Self::Output<Layer::Output, RecGradient>,
        combine_gradients: impl Fn(Layer::Gradient, RecGradient) -> NetworkGradient,
    ) -> Self::Output<Input, NetworkGradient> {
        let (epsilon_in, rec_gradient) = rec_opt_output;

        let epsilon_out = layer.backprop_layer(input, intermediary, &epsilon_in);
        let layer_gradient = layer.get_gradient(input, intermediary, &epsilon_in);

        (epsilon_out, combine_gradients(layer_gradient, rec_gradient))
    }
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;

    use super::*;
    use crate::{
        derivable::{activation::Tanh, loss::Euclidean, NeuraDerivable},
        prelude::*,
        utils::uniform_vector,
    };

    #[test]
    fn test_backprop_epsilon_bias() {
        // Checks that the epsilon term from backpropagation is well applied, by inspecting the bias terms
        // of the neural network's gradient

        for _ in 0..100 {
            let network = neura_sequential![
                neura_layer!("dense", 4, f64).activation(Tanh),
                neura_layer!("dense", 2, f64).activation(Tanh)
            ]
            .construct(NeuraShape::Vector(4))
            .unwrap();

            let optimizer = NeuraBackprop::new(Euclidean);
            let input = uniform_vector(4);
            let target = uniform_vector(2);

            let layer1_intermediary = &network.layer.weights * &input;
            let layer2_intermediary =
                &network.child_network.layer.weights * layer1_intermediary.map(|x| x.tanh());

            assert_relative_eq!(
                layer1_intermediary.map(|x| x.tanh()),
                network.clone().trim_tail().eval(&input)
            );

            let output = network.eval(&input);

            let gradient = optimizer.get_gradient(&network, &input, &target);

            let mut delta2_expected = Euclidean.nabla(&target, &output);
            for i in 0..2 {
                delta2_expected[i] *= Tanh.derivate(layer2_intermediary[i]);
            }
            let delta2_actual = gradient.1 .0 .1;

            assert_relative_eq!(delta2_actual.as_slice(), delta2_expected.as_slice());

            let gradient2_expected =
                &delta2_expected * layer1_intermediary.map(|x| x.tanh()).transpose();
            let gradient2_actual = gradient.1 .0 .0;

            assert_relative_eq!(gradient2_actual.as_slice(), gradient2_expected.as_slice());

            let mut delta1_expected =
                network.child_network.layer.weights.transpose() * delta2_expected;
            for i in 0..4 {
                delta1_expected[i] *= Tanh.derivate(layer1_intermediary[i]);
            }
            let delta1_actual = gradient.0 .1;

            assert_relative_eq!(delta1_actual.as_slice(), delta1_expected.as_slice());

            let gradient1_expected = &delta1_expected * input.transpose();
            let gradient1_actual = gradient.0 .0;

            assert_relative_eq!(gradient1_actual.as_slice(), gradient1_expected.as_slice());
        }
    }
}
