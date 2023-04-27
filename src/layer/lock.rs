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

impl<Input, Layer: NeuraLayer<Input>> NeuraLayer<Input> for NeuraLockLayer<Layer> {
    type Output = Layer::Output;

    fn eval(&self, input: &Input) -> Self::Output {
        self.layer.eval(input)
    }
}

impl<Layer: NeuraTrainableLayerBase> NeuraTrainableLayerBase for NeuraLockLayer<Layer> {
    type Gradient = ();

    fn default_gradient(&self) -> Self::Gradient {
        ()
    }

    fn apply_gradient(&mut self, _gradient: &Self::Gradient) {
        // Noop
    }
}

impl<Input, Layer: NeuraTrainableLayerEval<Input>> NeuraTrainableLayerEval<Input>
    for NeuraLockLayer<Layer>
{
    type IntermediaryRepr = Layer::IntermediaryRepr;

    fn eval_training(&self, input: &Input) -> (Self::Output, Self::IntermediaryRepr) {
        self.layer.eval_training(input)
    }
}

impl<Input, Layer: NeuraTrainableLayerEval<Input>> NeuraTrainableLayerSelf<Input>
    for NeuraLockLayer<Layer>
{
    fn regularize_layer(&self) -> Self::Gradient {
        ()
    }

    fn get_gradient(
        &self,
        _input: &Input,
        _intermediary: &Self::IntermediaryRepr,
        _epsilon: &Self::Output,
    ) -> Self::Gradient {
        ()
    }
}

impl<Input, Layer: NeuraTrainableLayerBackprop<Input>> NeuraTrainableLayerBackprop<Input>
    for NeuraLockLayer<Layer>
{
    fn backprop_layer(
        &self,
        input: &Input,
        intermediary: &Self::IntermediaryRepr,
        epsilon: &Self::Output,
    ) -> Input {
        self.layer.backprop_layer(input, intermediary, epsilon)
    }
}
