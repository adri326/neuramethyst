use num::ToPrimitive;

use crate::{derivable::NeuraLoss, layer::*, network::*};

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
        Trainable: NeuraLayer<Input> + NeuraNetworkRec,
        Loss: NeuraLoss<Trainable::Output, Target = Target> + Clone,
    > NeuraGradientSolver<Input, Target, Trainable> for NeuraBackprop<Loss>
where
    <Loss as NeuraLoss<Trainable::Output>>::Output: ToPrimitive,
    // Trainable: NeuraOldTrainableNetworkBase<Input, Gradient = <Trainable as NeuraLayerBase>::Gradient>,
    // Trainable: for<'a> NeuraOldTrainableNetwork<Input, (&'a NeuraBackprop<Loss>, &'a Target)>,
    for<'a> (&'a NeuraBackprop<Loss>, &'a Target):
        BackpropRecurse<Input, Trainable, <Trainable as NeuraLayerBase>::Gradient>,
{
    fn get_gradient(
        &self,
        trainable: &Trainable,
        input: &Input,
        target: &Target,
    ) -> <Trainable as NeuraLayerBase>::Gradient {
        let (_, gradient) = (self, target).recurse(trainable, input);
        // let (_, gradient) = trainable.traverse(input, &(self, target));

        gradient
    }

    fn score(&self, trainable: &Trainable, input: &Input, target: &Target) -> f64 {
        let output = trainable.eval(&input);
        self.loss.eval(target, &output).to_f64().unwrap()
    }
}

trait BackpropRecurse<Input, Network, Gradient> {
    fn recurse(&self, network: &Network, input: &Input) -> (Input, Gradient);
}

impl<Input, Loss: NeuraLoss<Input>> BackpropRecurse<Input, (), ()>
    for (&NeuraBackprop<Loss>, &Loss::Target)
{
    fn recurse(&self, _network: &(), input: &Input) -> (Input, ()) {
        (self.0.loss.nabla(self.1, input), ())
    }
}

impl<
        Input: Clone,
        Network: NeuraNetworkRec + NeuraNetwork<Input> + NeuraLayer<Input>,
        Loss,
        Target,
    > BackpropRecurse<Input, Network, Network::Gradient> for (&NeuraBackprop<Loss>, &Target)
where
    // Verify that we can traverse recursively
    for<'a> (&'a NeuraBackprop<Loss>, &'a Target): BackpropRecurse<
        Network::NodeOutput,
        Network::NextNode,
        <Network::NextNode as NeuraLayerBase>::Gradient,
    >,
    // Verify that the current layer implements the right traits
    Network::Layer: NeuraLayer<Network::LayerInput>,
    // Verify that the layer output can be cloned
    <Network::Layer as NeuraLayer<Network::LayerInput>>::Output: Clone,
    Network::NextNode: NeuraLayer<Network::NodeOutput>,
{
    fn recurse(&self, network: &Network, input: &Input) -> (Input, Network::Gradient) {
        let layer = network.get_layer();
        // Get layer output
        let layer_input = network.map_input(input);
        let (layer_output, layer_intermediary) = layer.eval_training(layer_input.as_ref());
        let output = network.map_output(input, &layer_output);

        // Recurse
        let (epsilon_in, gradient_rec) = self.recurse(network.get_next(), output.as_ref());

        // Get layer outgoing gradient vector
        let layer_epsilon_in = network.map_gradient_in(input, &epsilon_in);
        let layer_epsilon_out =
            layer.backprop_layer(&layer_input, &layer_intermediary, &layer_epsilon_in);
        let epsilon_out = network.map_gradient_out(input, &epsilon_in, &layer_epsilon_out);

        // Get layer parameter gradient
        let gradient = layer.get_gradient(&layer_input, &layer_intermediary, &layer_epsilon_in);

        (
            epsilon_out.into_owned(),
            network.merge_gradient(gradient_rec, gradient),
        )
    }
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use nalgebra::dvector;

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

    /// Check that there is no recursion error when using `()` in `recurse`
    #[test]
    fn test_recurse() {
        let backprop = NeuraBackprop::new(Euclidean);
        let target = dvector![0.0];

        (&backprop, &target).recurse(&(), &dvector![0.0]);
    }

    #[test]
    fn test_recurse_sequential() {
        let backprop = NeuraBackprop::new(Euclidean);
        let target = dvector![0.0];

        let network = neura_sequential![neura_layer!("dense", 4), neura_layer!("dense", 1),]
            .construct(NeuraShape::Vector(1))
            .unwrap();

        (&backprop, &target).recurse(&network, &dvector![0.0]);
    }
}
