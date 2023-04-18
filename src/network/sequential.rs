use num::Float;

use crate::{
    derivable::NeuraLoss,
    layer::{NeuraLayer, NeuraTrainableLayer, NeuraShape, NeuraPartialLayer},
};

use super::NeuraTrainableNetwork;

#[derive(Clone, Debug)]
pub struct NeuraSequential<Layer, ChildNetwork> {
    pub layer: Layer,
    pub child_network: Box<ChildNetwork>,
}

/// Operations on the tail end of a sequential network
pub trait NeuraSequentialTail {
    type TailTrimmed;
    type TailPushed<T>;

    fn trim_tail(self) -> Self::TailTrimmed;
    fn push_tail<T>(self, layer: T) -> Self::TailPushed<T>;
}

impl<Layer, ChildNetwork> NeuraSequential<Layer, ChildNetwork> {
    pub fn new(layer: Layer, child_network: ChildNetwork) -> Self {
        Self {
            layer,
            child_network: Box::new(child_network),
        }
    }

    pub fn new_match_output<Input>(layer: Layer, child_network: ChildNetwork) -> Self
    where
        Layer: NeuraLayer<Input>,
        ChildNetwork: NeuraLayer<Layer::Output>,
    {
        Self::new(layer, child_network)
    }

    pub fn trim_front(self) -> ChildNetwork {
        *self.child_network
    }

    pub fn push_front<Input, Input2, T: NeuraLayer<Input2, Output=Input>>(self, layer: T) -> NeuraSequential<T, Self>
    where
        Layer: NeuraLayer<Input>
    {
        NeuraSequential {
            layer: layer,
            child_network: Box::new(self),
        }
    }
}

// Trimming the last layer returns an empty network
impl<Layer> NeuraSequentialTail for NeuraSequential<Layer, ()> {
    type TailTrimmed = ();
    type TailPushed<T> = NeuraSequential<Layer, NeuraSequential<T, ()>>;

    fn trim_tail(self) -> Self::TailTrimmed {
        ()
    }

    fn push_tail<T>(self, layer: T) -> Self::TailPushed<T> {
        NeuraSequential {
            layer: self.layer,
            child_network: Box::new(NeuraSequential {
                layer,
                child_network: Box::new(()),
            }),
        }
    }
}

// Trimming another layer returns a network which calls trim recursively
impl<Layer, ChildNetwork: NeuraSequentialTail> NeuraSequentialTail
    for NeuraSequential<Layer, ChildNetwork>
{
    type TailTrimmed = NeuraSequential<Layer, <ChildNetwork as NeuraSequentialTail>::TailTrimmed>;
    type TailPushed<T> =
        NeuraSequential<Layer, <ChildNetwork as NeuraSequentialTail>::TailPushed<T>>;

    fn trim_tail(self) -> Self::TailTrimmed {
        NeuraSequential {
            layer: self.layer,
            child_network: Box::new(self.child_network.trim_tail()),
        }
    }

    fn push_tail<T>(self, layer: T) -> Self::TailPushed<T> {
        NeuraSequential {
            layer: self.layer,
            child_network: Box::new(self.child_network.push_tail(layer)),
        }
    }
}

impl<Input, Layer: NeuraLayer<Input>, ChildNetwork: NeuraLayer<Layer::Output>> NeuraLayer<Input>
    for NeuraSequential<Layer, ChildNetwork>
{
    type Output = ChildNetwork::Output;

    fn eval(&self, input: &Input) -> Self::Output {
        self.child_network.eval(&self.layer.eval(input))
    }
}

impl<Input: Clone> NeuraTrainableNetwork<Input> for () {
    type Delta = ();

    fn default_gradient(&self) -> () {
        ()
    }

    fn apply_gradient(&mut self, _gradient: &()) {
        // Noop
    }

    fn backpropagate<Loss: NeuraLoss<Input = Self::Output>>(
        &self,
        final_activation: &Input,
        target: &Loss::Target,
        loss: Loss,
    ) -> (Input, Self::Delta) {
        let backprop_epsilon = loss.nabla(target, &final_activation);

        (backprop_epsilon, ())
    }

    fn regularize(&self) -> () {
        ()
    }

    fn prepare(&mut self, _is_training: bool) {
        // Noop
    }
}

impl<Input, Layer: NeuraTrainableLayer<Input>, ChildNetwork: NeuraTrainableNetwork<Layer::Output>>
    NeuraTrainableNetwork<Input> for NeuraSequential<Layer, ChildNetwork>
{
    type Delta = (Layer::Gradient, Box<ChildNetwork::Delta>);

    fn default_gradient(&self) -> Self::Delta {
        (self.layer.default_gradient(), Box::new(self.child_network.default_gradient()))
    }

    fn apply_gradient(&mut self, gradient: &Self::Delta) {
        self.layer.apply_gradient(&gradient.0);
        self.child_network.apply_gradient(&gradient.1);
    }

    fn backpropagate<Loss: NeuraLoss<Input = Self::Output>>(
        &self,
        input: &Input,
        target: &Loss::Target,
        loss: Loss,
    ) -> (Input, Self::Delta) {
        let next_activation = self.layer.eval(input);
        let (backprop_gradient, weights_gradient) =
            self.child_network
                .backpropagate(&next_activation, target, loss);
        let (backprop_gradient, layer_gradient) =
            self.layer.backprop_layer(input, backprop_gradient);

        (
            backprop_gradient,
            (layer_gradient, Box::new(weights_gradient)),
        )
    }

    fn regularize(&self) -> Self::Delta {
        (
            self.layer.regularize_layer(),
            Box::new(self.child_network.regularize()),
        )
    }

    fn prepare(&mut self, is_training: bool) {
        self.layer.prepare_layer(is_training);
        self.child_network.prepare(is_training);
    }
}

impl<Layer> From<Layer> for NeuraSequential<Layer, ()> {
    fn from(layer: Layer) -> Self {
        Self {
            layer,
            child_network: Box::new(()),
        }
    }
}

pub trait NeuraSequentialBuild {
    type Constructed;
    type Err;

    fn construct(self, input_shape: NeuraShape) -> Result<Self::Constructed, Self::Err>;
}

#[derive(Debug, Clone)]
pub enum NeuraSequentialBuildErr<Err, ChildErr> {
    Current(Err),
    Child(ChildErr),
}

impl<Layer: NeuraPartialLayer> NeuraSequentialBuild for NeuraSequential<Layer, ()> {
    type Constructed = NeuraSequential<Layer::Constructed, ()>;
    type Err = Layer::Err;

    fn construct(self, input_shape: NeuraShape) -> Result<Self::Constructed, Self::Err> {
        Ok(NeuraSequential {
            layer: self.layer.construct(input_shape)?,
            child_network: Box::new(())
        })
    }
}

impl<Layer: NeuraPartialLayer + , ChildNetwork: NeuraSequentialBuild> NeuraSequentialBuild for NeuraSequential<Layer, ChildNetwork> {
    type Constructed = NeuraSequential<Layer::Constructed, ChildNetwork::Constructed>;
    type Err = NeuraSequentialBuildErr<Layer::Err, ChildNetwork::Err>;

    fn construct(self, input_shape: NeuraShape) -> Result<Self::Constructed, Self::Err> {
        let layer = self.layer.construct(input_shape).map_err(|e| NeuraSequentialBuildErr::Current(e))?;

        // TODO: ensure that this operation (and all recursive operations) are directly allocated on the heap
        let child_network = self.child_network
            .construct(Layer::output_shape(&layer))
            .map_err(|e| NeuraSequentialBuildErr::Child(e))?;
        let child_network = Box::new(child_network);

        Ok(NeuraSequential {
            layer,
            child_network,
        })
    }


}

/// An utility to recursively create a NeuraSequential network, while writing it in a declarative and linear fashion.
/// Note that this can quickly create big and unwieldly types.
#[macro_export]
macro_rules! neura_sequential {
    [] => {
        ()
    };

    [ $layer:expr $(,)? ] => {
        $crate::network::sequential::NeuraSequential::from($layer)
    };

    [ $first:expr, $($rest:expr),+ $(,)? ] => {
        $crate::network::sequential::NeuraSequential::new($first, neura_sequential![$($rest),+])
    };
}

#[cfg(test)]
mod test {
    use nalgebra::dvector;

    use crate::{
        derivable::{activation::Relu, regularize::NeuraL0},
        layer::{NeuraDenseLayer, NeuraShape, NeuraLayer},
        neura_layer,
    };

    use super::NeuraSequentialBuild;

    #[test]
    fn test_neura_network_macro() {
        let mut rng = rand::thread_rng();

        let _ = neura_sequential![
            NeuraDenseLayer::from_rng(8, 12, &mut rng, Relu, NeuraL0) as NeuraDenseLayer<f64, _, _>,
            NeuraDenseLayer::from_rng(12, 16, &mut rng, Relu, NeuraL0) as NeuraDenseLayer<f64, _, _>,
            NeuraDenseLayer::from_rng(16, 2, &mut rng, Relu, NeuraL0) as NeuraDenseLayer<f64, _, _>
        ];

        let _ = neura_sequential![
            NeuraDenseLayer::from_rng(2, 2, &mut rng, Relu, NeuraL0) as NeuraDenseLayer<f64, _, _>,
        ];

        let _ = neura_sequential![
            NeuraDenseLayer::from_rng(8, 16, &mut rng, Relu, NeuraL0) as NeuraDenseLayer<f64, _, _>,
            NeuraDenseLayer::from_rng(16, 12, &mut rng, Relu, NeuraL0) as NeuraDenseLayer<f64, _, _>,
        ];

        let network = neura_sequential![
            neura_layer!("dense", 16, Relu),
            neura_layer!("dense", 12, Relu),
            neura_layer!("dense", 2, Relu)
        ].construct(NeuraShape::Vector(2)).unwrap();

        network.eval(&dvector![0.0f64, 0.0]);
    }
}
