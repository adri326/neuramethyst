use dyn_clone::DynClone;
use std::{any::Any, fmt::Debug};

use crate::{
    algebra::NeuraDynVectorSpace,
    axis::{NeuraAxis, NeuraAxisDefault},
    prelude::{NeuraPartialLayer, NeuraShape},
};

use super::*;

pub trait NeuraGraphNodePartial<Data>: DynClone + Debug {
    fn inputs<'a>(&'a self) -> &'a [String];
    fn name<'a>(&'a self) -> &'a str;

    fn construct(
        &self,
        input_shapes: Vec<NeuraShape>,
    ) -> Result<(Box<dyn NeuraGraphNodeEval<Data>>, NeuraShape), String>;
}

pub trait NeuraGraphNodeEval<Data>: DynClone + Debug {
    fn eval(&self, inputs: &[Data]) -> Data;

    fn eval_training(&self, inputs: &[Data]) -> (Data, Box<dyn Any>);
    fn backprop(&self, intermediary: &dyn Any, epsilon_in: &Data) -> Vec<Data>;

    fn default_gradient(&self) -> Box<dyn NeuraDynVectorSpace>;

    fn get_gradient(
        &self,
        intermediary: &dyn Any,
        epsilon_in: &Data,
    ) -> Box<dyn NeuraDynVectorSpace>;

    fn get_regularization_gradient(&self) -> Box<dyn NeuraDynVectorSpace>;

    fn apply_gradient(&mut self, gradient: &dyn NeuraDynVectorSpace);

    fn prepare(&mut self, is_training: bool);
}

#[derive(Clone, Debug)]
pub struct NeuraGraphNode<Axis, Layer> {
    inputs: Vec<String>,
    axis: Axis,
    layer: Layer,
    name: String,

    input_shapes: Option<Vec<NeuraShape>>,
}

impl<Layer> NeuraGraphNode<NeuraAxisDefault, Layer> {
    pub(crate) fn from_layer(layer: Layer, input_shapes: Vec<NeuraShape>) -> Self {
        Self {
            inputs: vec![],
            axis: NeuraAxisDefault,
            layer,
            name: random_name(),

            input_shapes: Some(input_shapes),
        }
    }
}

impl<Axis, Layer> NeuraGraphNode<Axis, Layer> {
    pub fn new(inputs: Vec<String>, axis: Axis, layer: Layer, name: String) -> Self {
        // Check that `name not in inputs` ?
        Self {
            inputs,
            axis,
            layer,
            name,

            input_shapes: None,
        }
    }

    pub fn as_boxed<Data: Clone>(self) -> Box<dyn NeuraGraphNodePartial<Data>>
    where
        Axis: NeuraAxis<Data>,
        Layer: NeuraPartialLayer + Clone + Debug + 'static,
        Layer::Constructed: NeuraLayer<Axis::Combined, Output = Data>,
        Layer::Err: Debug,
        <Layer::Constructed as NeuraLayer<Axis::Combined>>::IntermediaryRepr: 'static,
    {
        Box::new(self)
    }

    fn downcast_intermediary<'a, Data>(
        &self,
        intermediary: &'a dyn Any,
    ) -> &'a Intermediary<Axis::Combined, Layer>
    where
        Axis: NeuraAxis<Data>,
        Layer: NeuraLayer<Axis::Combined>,
    {
        intermediary
            .downcast_ref::<Intermediary<Axis::Combined, Layer>>()
            .expect("Incompatible value passed to NeuraGraphNode::backprop")
    }
}

struct Intermediary<Combined, Layer: NeuraLayer<Combined>>
where
    Layer::IntermediaryRepr: 'static,
{
    combined: Combined,
    layer_intermediary: Layer::IntermediaryRepr,
}

impl<Data: Clone, Axis: NeuraAxis<Data>, Layer: NeuraLayer<Axis::Combined, Output = Data>>
    NeuraGraphNodeEval<Data> for NeuraGraphNode<Axis, Layer>
{
    fn eval<'a>(&'a self, inputs: &[Data]) -> Data {
        let combined = self.axis.combine(inputs);
        self.layer.eval(&combined)
    }

    fn eval_training<'a>(&self, inputs: &[Data]) -> (Data, Box<dyn Any>) {
        let combined = self.axis.combine(inputs);
        let (result, layer_intermediary) = self.layer.eval_training(&combined);

        let intermediary: Intermediary<Axis::Combined, Layer> = Intermediary {
            combined,
            layer_intermediary,
        };

        (result, Box::new(intermediary))
    }

    fn backprop(&self, intermediary: &dyn Any, epsilon_in: &Data) -> Vec<Data> {
        let intermediary = self.downcast_intermediary(intermediary);

        let epsilon_out = self.layer.backprop_layer(
            &intermediary.combined,
            &intermediary.layer_intermediary,
            epsilon_in,
        );

        self.axis
            .split(&epsilon_out, self.input_shapes.as_ref().unwrap())
    }

    fn get_gradient(
        &self,
        intermediary: &dyn Any,
        epsilon_in: &Data,
    ) -> Box<dyn NeuraDynVectorSpace> {
        let intermediary = self.downcast_intermediary(intermediary);

        Box::new(self.layer.get_gradient(
            &intermediary.combined,
            &intermediary.layer_intermediary,
            epsilon_in,
        ))
    }

    fn apply_gradient(&mut self, gradient: &dyn NeuraDynVectorSpace) {
        self.layer.apply_gradient(
            gradient
                .into_any()
                .downcast_ref::<Layer::Gradient>()
                .expect("Invalid gradient type passed to NeuraGraphNode::apply_gradient"),
        );
    }

    fn default_gradient(&self) -> Box<dyn NeuraDynVectorSpace> {
        Box::new(self.layer.default_gradient())
    }

    fn prepare(&mut self, is_training: bool) {
        self.layer.prepare_layer(is_training);
    }

    fn get_regularization_gradient(&self) -> Box<dyn NeuraDynVectorSpace> {
        Box::new(self.layer.regularize_layer())
    }
}

impl<Data: Clone, Axis: NeuraAxis<Data>, Layer: NeuraPartialLayer + Clone + Debug>
    NeuraGraphNodePartial<Data> for NeuraGraphNode<Axis, Layer>
where
    Layer::Constructed: NeuraLayer<Axis::Combined, Output = Data>,
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
            .shape(&input_shapes)
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
                input_shapes: Some(input_shapes),
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
