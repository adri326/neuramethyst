use crate::err::*;
use crate::layer::*;
use crate::network::*;
use crate::utils::unwrap_or_clone;

use NeuraResidualConstructErr::*;

use std::borrow::Cow;

use super::construct::*;
use super::*;

#[derive(Clone, Debug, PartialEq)]
pub struct NeuraResidualLast {
    output_shape: Option<NeuraShape>,
}

impl NeuraResidualLast {
    #[inline(always)]
    pub fn new() -> Self {
        Self { output_shape: None }
    }
}

impl Default for NeuraResidualLast {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl NeuraResidualConstruct for NeuraResidualLast {
    type Constructed = NeuraResidualLast;
    type Err = NeuraRecursiveErr<NeuraResidualConstructErr<(), NeuraAxisErr>, ()>;

    fn construct_residual(
        self,
        input: NeuraResidualInput<NeuraShape>,
        indices: NeuraResidualInput<usize>,
        current_index: usize,
    ) -> Result<Self::Constructed, Self::Err> {
        let (this_input, _rest) = input.shift();
        let index = indices
            .get_first()
            .ok_or(Self::Err::Current(AxisErr(NeuraAxisErr::NoInput)))?;

        if *index != current_index - 1 {
            return Err(Self::Err::Current(WrongConnection(
                current_index as isize - *index as isize - 1,
            )));
        }
        if this_input.len() != 1 {
            return Err(Self::Err::Current(AxisErr(NeuraAxisErr::NoInput)));
        }

        // TODO: check that rest contains nothing else

        let input = unwrap_or_clone(this_input.into_iter().next().unwrap());

        Ok(Self {
            output_shape: Some(input),
        })
    }
}

impl NeuraLayerBase for NeuraResidualLast {
    type Gradient = ();

    fn default_gradient(&self) -> Self::Gradient {
        
    }

    fn output_shape(&self) -> NeuraShape {
        self.output_shape
            .expect("Called NeuraResidualLast::output_shape before constructing it")
    }
}

impl NeuraNetworkBase for NeuraResidualLast {
    type Layer = ();

    #[inline(always)]
    fn get_layer(&self) -> &Self::Layer {
        &()
    }
}

impl NeuraNetworkRec for NeuraResidualLast {
    type NextNode = ();

    #[inline(always)]
    fn get_next(&self) -> &Self::NextNode {
        &()
    }

    #[inline(always)]
    fn merge_gradient(&self, _rec_gradient: (), _layer_gradient: ()) -> Self::Gradient
    where
        Self::Layer: NeuraLayerBase,
    {
        
    }
}

impl<Data: Clone> NeuraNetwork<NeuraResidualInput<Data>> for NeuraResidualLast {
    type LayerInput = Data;

    type NodeOutput = Data;

    fn map_input<'a>(&'_ self, input: &'a NeuraResidualInput<Data>) -> Cow<'a, Self::LayerInput> {
        Cow::Owned(unwrap_or_clone(input.clone().get_first().unwrap()))
    }

    fn map_output<'a>(
        &'_ self,
        _input: &'_ NeuraResidualInput<Data>,
        layer_output: &'a Data,
    ) -> Cow<'a, Self::NodeOutput> {
        Cow::Borrowed(layer_output)
    }

    fn map_gradient_in<'a>(
        &'_ self,
        _input: &'_ NeuraResidualInput<Data>,
        gradient_in: &'a Self::NodeOutput,
    ) -> Cow<'a, <Self::Layer as NeuraLayer<Self::LayerInput>>::Output> {
        Cow::Borrowed(gradient_in)
    }

    fn map_gradient_out<'a>(
        &'_ self,
        _input: &'_ NeuraResidualInput<Data>,
        _gradient_in: &'_ Self::NodeOutput,
        gradient_out: &'a Self::LayerInput,
    ) -> Cow<'a, NeuraResidualInput<Data>> {
        let mut result = NeuraResidualInput::new();

        result.push(0, Rc::new(gradient_out.clone()));

        Cow::Owned(result)
    }
}

impl<Data: Clone> NeuraLayer<NeuraResidualInput<Data>> for NeuraResidualLast {
    type Output = Data;
    type IntermediaryRepr = ();

    fn eval_training(&self, input: &NeuraResidualInput<Data>) -> (Self::Output, ()) {
        let result: Rc<Self::Output> = input.clone().get_first()
            .expect("Invalid NeuraResidual state: network returned no data, did you forget to link the last layer?");

        (unwrap_or_clone(result), ())
    }

    fn backprop_layer(
        &self,
        input: &NeuraResidualInput<Data>,
        _intermediary: &Self::IntermediaryRepr,
        epsilon: &Self::Output,
    ) -> NeuraResidualInput<Data> {
        Cow::into_owned(self.map_gradient_out(input, epsilon, epsilon))
    }
}
