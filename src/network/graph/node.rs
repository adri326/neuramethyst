use dyn_clone::DynClone;
use std::fmt::Debug;

use crate::{
    err::NeuraAxisErr,
    layer::{NeuraLayer, NeuraShapedLayer},
    network::residual::{NeuraAxisDefault, NeuraCombineInputs, NeuraSplitInputs},
    prelude::{NeuraPartialLayer, NeuraShape},
};

// TODO: split into two  traits
pub trait NeuraGraphNodePartial<Data>: DynClone + Debug {
    fn inputs<'a>(&'a self) -> &'a [String];
    fn name<'a>(&'a self) -> &'a str;

    fn construct(
        &self,
        input_shapes: Vec<NeuraShape>,
    ) -> Result<(Box<dyn NeuraGraphNodeEval<Data>>, NeuraShape), String>;
}

pub trait NeuraGraphNodeEval<Data>: DynClone + Debug {
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
            + Debug
            + 'static,
        Layer: NeuraPartialLayer + Clone + Debug + 'static,
        Layer::Constructed: NeuraShapedLayer
            + NeuraLayer<<Axis as NeuraCombineInputs<Data>>::Combined, Output = Data>
            + Clone
            + Debug
            + 'static,
        Layer::Err: Debug,
    {
        Box::new(self)
    }
}

impl<
        Data: Clone,
        Axis: NeuraSplitInputs<Data> + Clone + Debug,
        Layer: NeuraLayer<<Axis as NeuraCombineInputs<Data>>::Combined, Output = Data> + Clone + Debug,
    > NeuraGraphNodeEval<Data> for NeuraGraphNode<Axis, Layer>
{
    fn eval<'a>(&'a self, inputs: &[Data]) -> Data {
        // TODO: use to_vec_in?
        let combined = self.axis.combine(inputs.to_vec());
        self.layer.eval(&combined)
    }
}

impl<Layer: Clone + Debug> From<Layer> for NeuraGraphNode<NeuraAxisDefault, Layer> {
    fn from(layer: Layer) -> Self {
        Self {
            inputs: vec![],
            axis: NeuraAxisDefault,
            layer,
            name: random_name(),
        }
    }
}

impl<
        Data: Clone,
        Axis: NeuraSplitInputs<Data>
            + NeuraCombineInputs<NeuraShape, Combined = Result<NeuraShape, NeuraAxisErr>>
            + Clone
            + Debug
            + 'static,
        Layer: NeuraPartialLayer + Clone + Debug,
    > NeuraGraphNodePartial<Data> for NeuraGraphNode<Axis, Layer>
where
    Layer::Constructed: NeuraShapedLayer
        + NeuraLayer<<Axis as NeuraCombineInputs<Data>>::Combined, Output = Data>
        + Clone
        + Debug
        + 'static,
    Layer::Err: Debug,
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

pub fn random_name() -> String {
    use rand::Rng;
    use std::fmt::Write;

    let mut res = String::with_capacity(10);
    write!(&mut res, "value_").unwrap();

    let mut rng = rand::thread_rng();

    for _ in 0..4 {
        let ch = char::from_u32(rng.gen_range((b'a' as u32)..(b'z' as u32))).unwrap();
        write!(&mut res, "{}", ch).unwrap();
    }

    res
}
