use super::*;

/// A layer wrapper that disables any kind of training for the wrappee:
/// traits like NeuraTrainableLayerBackprop will still work as-is,
/// but `apply_gradient` will do nothing, and weights gradient computation is skipped.
#[derive(Clone, Debug)]
pub struct NeuraLockLayer<Layer: ?Sized> {
    layer: Box<Layer>,
}

impl<Layer> NeuraLockLayer<Layer> {
    pub fn new(layer: Layer) -> Self {
        Self {
            layer: Box::new(layer),
        }
    }

    pub fn unlock_layer(self) -> Layer {
        *self.layer
    }

    pub fn get(&self) -> &Layer {
        &self.layer
    }
}

impl<Layer: NeuraLayerBase> NeuraLayerBase for NeuraLockLayer<Layer> {
    type Gradient = ();

    fn output_shape(&self) -> NeuraShape {
        self.layer.output_shape()
    }

    fn default_gradient(&self) -> Self::Gradient {
        
    }

    fn prepare_layer(&mut self, is_training: bool) {
        self.layer.prepare_layer(is_training);
    }
}

impl<Input, Layer: NeuraLayer<Input>> NeuraLayer<Input> for NeuraLockLayer<Layer> {
    type Output = Layer::Output;

    fn eval(&self, input: &Input) -> Self::Output {
        self.layer.eval(input)
    }

    type IntermediaryRepr = Layer::IntermediaryRepr;

    fn eval_training(&self, input: &Input) -> (Self::Output, Self::IntermediaryRepr) {
        self.layer.eval_training(input)
    }

    fn backprop_layer(
        &self,
        input: &Input,
        intermediary: &Self::IntermediaryRepr,
        epsilon: &Self::Output,
    ) -> Input {
        self.layer.backprop_layer(input, intermediary, epsilon)
    }
}
