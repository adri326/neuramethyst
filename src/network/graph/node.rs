use crate::{layer::NeuraLayer, network::residual::NeuraSplitInputs};

pub trait NeuraGraphNodeTrait<Data> {
    fn eval<'a>(&'a self, inputs: &[Data]) -> Data;

    fn inputs<'a>(&'a self) -> &'a [String];
    fn name<'a>(&'a self) -> &'a str;
}

pub struct NeuraGraphNode<Axis, Layer> {
    inputs: Vec<String>,
    axis: Axis,
    layer: Layer,
    name: String,
}

impl<Axis, Layer> NeuraGraphNode<Axis, Layer> {
    pub fn new(inputs: Vec<String>, axis: Axis, layer: Layer, name: String) -> Self {
        // Check that `name not in inputs` ?
        Self {
            inputs,
            axis,
            layer,
            name,
        }
    }

    pub fn as_boxed<Data: Clone>(self) -> Box<dyn NeuraGraphNodeTrait<Data>>
    where
        Axis: NeuraSplitInputs<Data> + 'static,
        Layer: NeuraLayer<Axis::Combined, Output = Data> + 'static,
    {
        Box::new(self)
    }
}

impl<
        Data: Clone,
        Axis: NeuraSplitInputs<Data>,
        Layer: NeuraLayer<Axis::Combined, Output = Data>,
    > NeuraGraphNodeTrait<Data> for NeuraGraphNode<Axis, Layer>
{
    fn eval<'a>(&'a self, inputs: &[Data]) -> Data {
        // TODO: use to_vec_in?
        let combined = self.axis.combine(inputs.to_vec());
        self.layer.eval(&combined)
    }

    fn inputs<'a>(&'a self) -> &'a [String] {
        &self.inputs
    }

    fn name<'a>(&'a self) -> &'a str {
        &self.name
    }
}
