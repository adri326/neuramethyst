use dyn_clone::DynClone;

use crate::{
    err::NeuraAxisErr,
    layer::{NeuraLayer, NeuraShapedLayer},
    network::residual::{NeuraCombineInputs, NeuraSplitInputs},
    prelude::{NeuraPartialLayer, NeuraShape},
};

// TODO: split into two  traits
pub trait NeuraGraphNodePartial<Data>: DynClone + std::fmt::Debug {
    fn inputs<'a>(&'a self) -> &'a [String];
    fn name<'a>(&'a self) -> &'a str;

    fn construct(
        &self,
        input_shapes: Vec<NeuraShape>,
    ) -> Result<(Box<dyn NeuraGraphNodeEval<Data>>, NeuraShape), String>;
}

pub trait NeuraGraphNodeEval<Data>: DynClone + std::fmt::Debug {
    fn eval<'a>(&'a self, inputs: &[Data]) -> Data;
}

#[derive(Clone, Debug)]
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

    pub fn as_boxed<Data: Clone>(self) -> Box<dyn NeuraGraphNodePartial<Data>>
    where
        Axis: NeuraSplitInputs<Data>
            + NeuraCombineInputs<NeuraShape, Combined = Result<NeuraShape, NeuraAxisErr>>
            + Clone
            + std::fmt::Debug
            + 'static,
        Layer: NeuraPartialLayer + Clone + std::fmt::Debug + 'static,
        Layer::Constructed: NeuraShapedLayer
            + NeuraLayer<<Axis as NeuraCombineInputs<Data>>::Combined, Output = Data>
            + Clone
            + std::fmt::Debug
            + 'static,
        Layer::Err: std::fmt::Debug,
    {
        Box::new(self)
    }
}

impl<
        Data: Clone,
        Axis: NeuraSplitInputs<Data> + Clone + std::fmt::Debug,
        Layer: NeuraLayer<<Axis as NeuraCombineInputs<Data>>::Combined, Output = Data>
            + Clone
            + std::fmt::Debug,
    > NeuraGraphNodeEval<Data> for NeuraGraphNode<Axis, Layer>
{
    fn eval<'a>(&'a self, inputs: &[Data]) -> Data {
        // TODO: use to_vec_in?
        let combined = self.axis.combine(inputs.to_vec());
        self.layer.eval(&combined)
    }
}

impl<
        Data: Clone,
        Axis: NeuraSplitInputs<Data>
            + NeuraCombineInputs<NeuraShape, Combined = Result<NeuraShape, NeuraAxisErr>>
            + Clone
            + std::fmt::Debug
            + 'static,
        Layer: NeuraPartialLayer + Clone + std::fmt::Debug,
    > NeuraGraphNodePartial<Data> for NeuraGraphNode<Axis, Layer>
where
    Layer::Constructed: NeuraShapedLayer
        + NeuraLayer<<Axis as NeuraCombineInputs<Data>>::Combined, Output = Data>
        + Clone
        + std::fmt::Debug
        + 'static,
    Layer::Err: std::fmt::Debug,
{
    fn inputs<'a>(&'a self) -> &'a [String] {
        &self.inputs
    }

    fn name<'a>(&'a self) -> &'a str {
        &self.name
    }

    fn construct(
        &self,
        input_shapes: Vec<NeuraShape>,
    ) -> Result<(Box<dyn NeuraGraphNodeEval<Data>>, NeuraShape), String> {
        let combined = self
            .axis
            .combine(input_shapes)
            .map_err(|err| format!("{:?}", err))?;

        let constructed_layer = self
            .layer
            .clone()
            .construct(combined)
            .map_err(|err| format!("{:?}", err))?;
        let output_shape = constructed_layer.output_shape();

        Ok((
            Box::new(NeuraGraphNode {
                inputs: self.inputs.clone(),
                axis: self.axis.clone(),
                layer: constructed_layer,
                name: self.name.clone(),
            }),
            output_shape,
        ))
    }
}
