use nalgebra::{DVector, Scalar};
use num::{traits::NumAssignOps, Float};

use crate::{
    derivable::NeuraDerivable,
    layer::*,
    network::{NeuraNetwork, NeuraNetworkRec},
    prelude::NeuraLayer,
};

use super::*;

// TODO: add `max_depth: usize`
pub struct NeuraForwardForward<Act> {
    threshold: f64,
    activation: Act,
}

impl<Act: Clone + NeuraDerivable<f64>> NeuraForwardForward<Act> {
    pub fn new(activation: Act, threshold: f64) -> Self {
        Self {
            threshold,
            activation,
        }
    }
}

trait ForwardForwardDerivate<Data> {
    fn derivate_goodness(&self, data: &Data) -> Data;
}

impl<F: Float + Scalar + NumAssignOps, Act: NeuraDerivable<F>> ForwardForwardDerivate<DVector<F>>
    for NeuraForwardPair<Act>
{
    fn derivate_goodness(&self, data: &DVector<F>) -> DVector<F> {
        let goodness = data
            .iter()
            .copied()
            .reduce(|acc, x| acc + x * x)
            .unwrap_or(F::zero());
        let goodness = if self.maximize {
            goodness - F::from(self.threshold).unwrap()
        } else {
            F::from(self.threshold).unwrap() - goodness
        };
        // We skip self.activation.eval(goodness)

        let two = F::from(2.0).unwrap();

        // The original formula does not have a 1/2 term,
        // so we must multiply by 2
        let mut goodness_derivative = data * (two * self.activation.derivate(goodness));

        if self.maximize {
            goodness_derivative = -goodness_derivative;
        }

        goodness_derivative
    }
}

struct NeuraForwardPair<Act> {
    threshold: f64,
    maximize: bool,
    activation: Act,
}

impl<
        F: Float,
        Act: Clone + NeuraDerivable<f64>,
        Input: Clone,
        Trainable: NeuraLayer<Input, Output = DVector<F>>,
    > NeuraGradientSolver<Input, bool, Trainable> for NeuraForwardForward<Act>
where
    NeuraForwardPair<Act>:
        ForwardForwardRecurse<Input, Trainable, <Trainable as NeuraLayerBase>::Gradient>,
{
    fn get_gradient(
        &self,
        trainable: &Trainable,
        input: &Input,
        target: &bool,
    ) -> <Trainable as NeuraLayerBase>::Gradient {
        let target = *target;
        let pair = NeuraForwardPair {
            threshold: self.threshold,
            maximize: target,
            activation: self.activation.clone(),
        };

        // trainable.traverse(
        //     input,
        //     &pair,
        // )

        pair.recurse(trainable, input)
    }

    fn score(&self, trainable: &Trainable, input: &Input, target: &bool) -> f64 {
        let output = trainable.eval(input);
        let goodness = output
            .iter()
            .map(|x| x.to_f64().unwrap())
            .reduce(|acc, x| acc + x * x)
            .unwrap_or(0.0);
        let goodness = goodness - self.threshold;
        let goodness = self.activation.eval(goodness);

        // Try to normalize the goodness so that if `Act([-threshold; +inf]) = [α; 1]`, `goodness ∈ [0; 1]`
        let activation_zero = self.activation.eval(-self.threshold);
        const EPSILON: f64 = 0.01;
        let goodness = if activation_zero < 1.0 - EPSILON {
            (goodness - activation_zero) / (1.0 - activation_zero)
        } else {
            goodness
        };

        if *target {
            1.0 - goodness
        } else {
            goodness
        }
    }
}

trait ForwardForwardRecurse<Input, Network, Gradient> {
    fn recurse(&self, network: &Network, input: &Input) -> Gradient;
}

impl<Act, Input> ForwardForwardRecurse<Input, (), ()> for NeuraForwardPair<Act> {
    #[inline(always)]
    fn recurse(&self, _network: &(), _input: &Input) {
        
    }
}

impl<Act, Input: Clone, Network: NeuraNetwork<Input> + NeuraNetworkRec>
    ForwardForwardRecurse<Input, Network, Network::Gradient> for NeuraForwardPair<Act>
where
    Network::Layer: NeuraLayer<Network::LayerInput>,
    <Network::Layer as NeuraLayer<Network::LayerInput>>::Output: Clone,
    Self: ForwardForwardDerivate<<Network::Layer as NeuraLayer<Network::LayerInput>>::Output>,
    Self: ForwardForwardRecurse<
        Network::NodeOutput,
        Network::NextNode,
        <Network::NextNode as NeuraLayerBase>::Gradient,
    >,
{
    fn recurse(&self, network: &Network, input: &Input) -> Network::Gradient {
        let layer = network.get_layer();
        let layer_input = network.map_input(input);
        let (layer_output, layer_intermediary) = layer.eval_training(&layer_input);
        let output = network.map_output(input, &layer_output);

        let derivative = self.derivate_goodness(&layer_output);

        let layer_gradient = layer.get_gradient(&layer_input, &layer_intermediary, &derivative);

        network.merge_gradient(self.recurse(network.get_next(), &output), layer_gradient)
    }
}

#[cfg(test)]
mod test {
    use rand::Rng;

    use super::*;
    use crate::{
        derivable::activation::{Relu, Tanh},
        prelude::*,
        utils::uniform_vector,
    };

    #[test]
    fn test_forward_forward() {
        let network = neura_sequential![
            neura_layer!("dense", 10).activation(Relu),
            neura_layer!("normalize"),
            neura_layer!("dense", 4).activation(Relu),
            neura_layer!("normalize"),
            neura_layer!("dense", 1),
        ]
        .construct(NeuraShape::Vector(10))
        .unwrap();

        let fwd_solver = NeuraForwardForward::new(Tanh, 0.25);

        fwd_solver.get_gradient(&network, &uniform_vector(10).map(|x| x as f32), &true);
    }

    #[test]
    fn test_forward_forward_train() {
        let mut network = neura_sequential![
            neura_layer!("dense", 5).activation(Relu),
            neura_layer!("normalize"),
            neura_layer!("dense", 3).activation(Relu),
            neura_layer!("normalize"),
            neura_layer!("dense", 1)
        ]
        .construct(NeuraShape::Vector(4))
        .unwrap();

        let solver = NeuraForwardForward::new(Tanh, 0.25);

        let trainer = NeuraBatchedTrainer::new()
            .learning_rate(0.01)
            .iterations(20);

        let inputs = (0..1).cycle().map(|_| {
            let mut rng = rand::thread_rng();
            (uniform_vector(4).map(|x| x as f32), rng.gen_bool(0.5))
        });

        let test_inputs: Vec<_> = inputs.clone().take(10).collect();

        trainer.train(&solver, &mut network, inputs, test_inputs.as_slice());
    }
}
