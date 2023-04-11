use crate::{layer::NeuraLayer, train::{NeuraTrainable, NeuraTrainableLayer}, derivable::NeuraLoss};

pub struct NeuraNetwork<Layer: NeuraLayer, ChildNetwork> {
    layer: Layer,
    child_network: ChildNetwork,
}

impl<Layer: NeuraLayer, ChildNetwork> NeuraNetwork<Layer, ChildNetwork> {
    pub fn new(layer: Layer, child_network: ChildNetwork) -> Self {
        Self {
            layer,
            child_network,
        }
    }

    pub fn new_match_output(layer: Layer, child_network: ChildNetwork) -> Self
    where
        ChildNetwork: NeuraLayer<Input = Layer::Output>,
    {
        Self::new(layer, child_network)
    }

    pub fn child_network(&self) -> &ChildNetwork {
        &self.child_network
    }

    pub fn layer(&self) -> &Layer {
        &self.layer
    }
}

impl<Layer: NeuraLayer> From<Layer> for NeuraNetwork<Layer, ()> {
    fn from(layer: Layer) -> Self {
        Self {
            layer,
            child_network: (),
        }
    }
}

impl<Layer: NeuraLayer> NeuraLayer for NeuraNetwork<Layer, ()> {
    type Input = Layer::Input;
    type Output = Layer::Output;

    fn eval(&self, input: &Self::Input) -> Self::Output {
        self.layer.eval(input)
    }
}

impl<Layer: NeuraLayer, ChildNetwork: NeuraLayer<Input = Layer::Output>> NeuraLayer
    for NeuraNetwork<Layer, ChildNetwork>
{
    type Input = Layer::Input;

    type Output = ChildNetwork::Output;

    fn eval(&self, input: &Self::Input) -> Self::Output {
        self.child_network.eval(&self.layer.eval(input))
    }
}

impl<Layer: NeuraTrainableLayer> NeuraTrainable for NeuraNetwork<Layer, ()> {
    type Delta = Layer::Delta;

    fn backpropagate<Loss: NeuraLoss<Self::Output>>(&self, input: &Self::Input, target: Loss::Target, loss: Loss) -> (Self::Input, Self::Delta) {
        let final_activation = self.layer.eval(input);
        let backprop_epsilon = loss.nabla(target, final_activation);
        self.layer.backpropagate(&input, backprop_epsilon)
    }
}

impl<Layer: NeuraTrainableLayer, ChildNetwork: NeuraTrainable<Input = Layer::Output>> NeuraTrainable for NeuraNetwork<Layer, ChildNetwork> {
    type Delta = (Layer::Delta, ChildNetwork::Delta);

    fn backpropagate<Loss: NeuraLoss<Self::Output>>(&self, input: &Self::Input, target: Loss::Target, loss: Loss) -> (Self::Input, Self::Delta) {
        let next_activation = self.layer.eval(input);
        let (backprop_gradient, weights_gradient) = self.child_network.backpropagate(&next_activation, target, loss);
        let (backprop_gradient, layer_gradient) = self.layer.backpropagate(input, backprop_gradient);

        (backprop_gradient, (layer_gradient, weights_gradient))
    }
}

#[macro_export]
macro_rules! neura_network {
    [] => {
        ()
    };

    [ $layer:expr $(,)? ] => {
        NeuraNetwork::from($layer)
    };

    [ $first:expr, $($rest:expr),+ $(,)? ] => {
        NeuraNetwork::new_match_output($first, neura_network![$($rest),+])
    };
}

#[cfg(test)]
mod test {
    use crate::{derivable::activation::Relu, layer::NeuraDenseLayer, neura_layer};

    use super::*;

    #[test]
    fn test_neura_network_macro() {
        let mut rng = rand::thread_rng();

        let _ = neura_network![
            NeuraDenseLayer::from_rng(&mut rng, Relu) as NeuraDenseLayer<_, 8, 16>,
            NeuraDenseLayer::from_rng(&mut rng, Relu) as NeuraDenseLayer<_, _, 12>,
            NeuraDenseLayer::from_rng(&mut rng, Relu) as NeuraDenseLayer<_, _, 2>
        ];

        let _ =
            neura_network![NeuraDenseLayer::from_rng(&mut rng, Relu) as NeuraDenseLayer<_, 8, 16>,];

        let _ = neura_network![
            NeuraDenseLayer::from_rng(&mut rng, Relu) as NeuraDenseLayer<_, 8, 16>,
            NeuraDenseLayer::from_rng(&mut rng, Relu) as NeuraDenseLayer<_, _, 12>,
        ];

        let _ = neura_network![
            neura_layer!("dense", Relu, 16, 8),
            neura_layer!("dense", Relu, 12),
            neura_layer!("dense", Relu, 2)
        ];
    }
}
