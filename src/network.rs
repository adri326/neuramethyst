use crate::{
    derivable::NeuraLoss,
    layer::NeuraLayer,
    train::{NeuraTrainable, NeuraTrainableLayer},
};

#[derive(Clone, Debug)]
pub struct NeuraNetwork<Layer: NeuraLayer, ChildNetwork> {
    pub layer: Layer,
    pub child_network: ChildNetwork,
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

    pub fn trim_front(self) -> ChildNetwork {
        self.child_network
    }

    pub fn push_front<T: NeuraLayer>(self, layer: T) -> NeuraNetwork<T, Self> {
        NeuraNetwork {
            layer: layer,
            child_network: self,
        }
    }
}

/// Operations on the tail end of the network
pub trait NeuraNetworkTail {
    type TailTrimmed;
    type TailPushed<T: NeuraLayer>;

    fn trim_tail(self) -> Self::TailTrimmed;
    fn push_tail<T: NeuraLayer>(self, layer: T) -> Self::TailPushed<T>;
}

// Trimming the last layer returns an empty network
impl<Layer: NeuraLayer> NeuraNetworkTail for NeuraNetwork<Layer, ()> {
    type TailTrimmed = ();
    type TailPushed<T: NeuraLayer> = NeuraNetwork<Layer, NeuraNetwork<T, ()>>;

    fn trim_tail(self) -> Self::TailTrimmed {
        ()
    }

    fn push_tail<T: NeuraLayer>(self, layer: T) -> Self::TailPushed<T> {
        NeuraNetwork {
            layer: self.layer,
            child_network: NeuraNetwork {
                layer,
                child_network: (),
            },
        }
    }
}

// Trimming another layer returns a network which calls trim recursively
impl<Layer: NeuraLayer, ChildNetwork: NeuraNetworkTail> NeuraNetworkTail
    for NeuraNetwork<Layer, ChildNetwork>
{
    type TailTrimmed = NeuraNetwork<Layer, <ChildNetwork as NeuraNetworkTail>::TailTrimmed>;
    type TailPushed<T: NeuraLayer> =
        NeuraNetwork<Layer, <ChildNetwork as NeuraNetworkTail>::TailPushed<T>>;

    fn trim_tail(self) -> Self::TailTrimmed {
        NeuraNetwork {
            layer: self.layer,
            child_network: self.child_network.trim_tail(),
        }
    }

    fn push_tail<T: NeuraLayer>(self, layer: T) -> Self::TailPushed<T> {
        NeuraNetwork {
            layer: self.layer,
            child_network: self.child_network.push_tail(layer),
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

    fn apply_gradient(&mut self, gradient: &Self::Delta) {
        self.layer.apply_gradient(gradient);
    }

    fn backpropagate<Loss: NeuraLoss<Input = Self::Output>>(
        &self,
        input: &Self::Input,
        target: &Loss::Target,
        loss: Loss,
    ) -> (Self::Input, Self::Delta) {
        let final_activation = self.layer.eval(input);
        let backprop_epsilon = loss.nabla(target, &final_activation);
        self.layer.backpropagate(&input, backprop_epsilon)
    }

    fn regularize(&self) -> Self::Delta {
        self.layer.regularize()
    }

    fn prepare_epoch(&mut self) {
        self.layer.prepare_epoch();
    }

    fn cleanup(&mut self) {
        self.layer.cleanup();
    }
}

impl<Layer: NeuraTrainableLayer, ChildNetwork: NeuraTrainable<Input = Layer::Output>> NeuraTrainable
    for NeuraNetwork<Layer, ChildNetwork>
{
    type Delta = (Layer::Delta, ChildNetwork::Delta);

    fn apply_gradient(&mut self, gradient: &Self::Delta) {
        self.layer.apply_gradient(&gradient.0);
        self.child_network.apply_gradient(&gradient.1);
    }

    fn backpropagate<Loss: NeuraLoss<Input = Self::Output>>(
        &self,
        input: &Self::Input,
        target: &Loss::Target,
        loss: Loss,
    ) -> (Self::Input, Self::Delta) {
        let next_activation = self.layer.eval(input);
        let (backprop_gradient, weights_gradient) =
            self.child_network
                .backpropagate(&next_activation, target, loss);
        let (backprop_gradient, layer_gradient) =
            self.layer.backpropagate(input, backprop_gradient);

        (backprop_gradient, (layer_gradient, weights_gradient))
    }

    fn regularize(&self) -> Self::Delta {
        (self.layer.regularize(), self.child_network.regularize())
    }

    fn prepare_epoch(&mut self) {
        self.layer.prepare_epoch();
        self.child_network.prepare_epoch();
    }

    fn cleanup(&mut self) {
        self.layer.cleanup();
        self.child_network.cleanup();
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

#[macro_export]
macro_rules! neura_network {
    [] => {
        ()
    };

    [ $layer:expr $(,)? ] => {
        $crate::network::NeuraNetwork::from($layer)
    };

    [ $first:expr, $($rest:expr),+ $(,)? ] => {
        $crate::network::NeuraNetwork::new_match_output($first, neura_network![$($rest),+])
    };
}

#[cfg(test)]
mod test {
    use crate::{
        derivable::{activation::Relu, regularize::NeuraL0},
        layer::NeuraDenseLayer,
        neura_layer,
    };

    #[test]
    fn test_neura_network_macro() {
        let mut rng = rand::thread_rng();

        let _ = neura_network![
            NeuraDenseLayer::from_rng(&mut rng, Relu, NeuraL0) as NeuraDenseLayer<_, _, 8, 16>,
            NeuraDenseLayer::from_rng(&mut rng, Relu, NeuraL0) as NeuraDenseLayer<_, _, _, 12>,
            NeuraDenseLayer::from_rng(&mut rng, Relu, NeuraL0) as NeuraDenseLayer<_, _, _, 2>
        ];

        let _ = neura_network![
            NeuraDenseLayer::from_rng(&mut rng, Relu, NeuraL0) as NeuraDenseLayer<_, _, 8, 16>,
        ];

        let _ = neura_network![
            NeuraDenseLayer::from_rng(&mut rng, Relu, NeuraL0) as NeuraDenseLayer<_, _, 8, 16>,
            NeuraDenseLayer::from_rng(&mut rng, Relu, NeuraL0) as NeuraDenseLayer<_, _, _, 12>,
        ];

        let _ = neura_network![
            neura_layer!("dense", 8, 16; Relu),
            neura_layer!("dense", 12; Relu),
            neura_layer!("dense", 2; Relu)
        ];
    }
}
