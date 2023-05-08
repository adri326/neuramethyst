use crate::err::*;

use super::*;
use NeuraResidualConstructErr::*;

pub trait NeuraResidualConstruct {
    type Constructed;
    type Err;

    fn construct_residual(
        self,
        inputs: NeuraResidualInput<NeuraShape>,
        indices: NeuraResidualInput<usize>,
        current_index: usize,
    ) -> Result<Self::Constructed, Self::Err>;
}

impl<Layer: NeuraPartialLayer, ChildNetwork: NeuraResidualConstruct, Axis> NeuraResidualConstruct
    for NeuraResidualNode<Layer, ChildNetwork, Axis>
where
    Axis: NeuraCombineInputs<NeuraShape, Combined = Result<NeuraShape, NeuraAxisErr>>,
{
    type Constructed = NeuraResidualNode<Layer::Constructed, ChildNetwork::Constructed, Axis>;
    type Err = NeuraRecursiveErr<NeuraResidualConstructErr<Layer::Err>, ChildNetwork::Err>;

    fn construct_residual(
        self,
        inputs: NeuraResidualInput<NeuraShape>,
        indices: NeuraResidualInput<usize>,
        current_index: usize,
    ) -> Result<Self::Constructed, Self::Err> {
        let (input_shapes, mut rest_inputs) = inputs.shift();
        let (this_indices, mut rest_indices) = indices.shift();

        let self_input_shapes = input_shapes.iter().map(|x| **x).collect::<Vec<_>>();

        let layer_input_shape = self
            .axis
            .combine(input_shapes)
            .map_err(|e| NeuraRecursiveErr::Current(AxisErr(e)))?;

        let layer = self
            .layer
            .construct(layer_input_shape)
            .map_err(|e| NeuraRecursiveErr::Current(Layer(e)))?;
        let layer_shape = Rc::new(layer.output_shape());

        if self.offsets.len() == 0 {
            return Err(NeuraRecursiveErr::Current(NoOutput));
        }

        for &offset in &self.offsets {
            rest_inputs.push(offset, Rc::clone(&layer_shape));
            rest_indices.push(offset, Rc::new(current_index));
        }
        let layer_shape = *layer_shape;

        debug_assert!(this_indices.iter().all(|x| **x < current_index));
        let input_offsets: Vec<usize> = this_indices
            .into_iter()
            .map(|x| current_index - *x - 1)
            .collect();

        let child_network = self
            .child_network
            .construct_residual(rest_inputs, rest_indices, current_index + 1)
            .map_err(|e| NeuraRecursiveErr::Child(e))?;

        Ok(NeuraResidualNode {
            layer,
            child_network,
            offsets: self.offsets,
            axis: self.axis,
            output_shape: Some(layer_shape),
            input_shapes: self_input_shapes,
            input_offsets,
        })
    }
}

impl<Layers: NeuraResidualConstruct> NeuraPartialLayer for NeuraResidual<Layers>
where
    // Should always be satisfied:
    Layers::Constructed: NeuraLayerBase,
{
    type Constructed = NeuraResidual<Layers::Constructed>;
    type Err = Layers::Err;

    fn construct(self, input_shape: NeuraShape) -> Result<Self::Constructed, Self::Err> {
        let input_shape = Rc::new(input_shape);
        let mut inputs = NeuraResidualInput::new();
        let mut indices = NeuraResidualInput::new();

        for &offset in &self.initial_offsets {
            inputs.push(offset, Rc::clone(&input_shape));
            indices.push(offset, Rc::new(0usize));
        }

        drop(input_shape);

        let layers = self.layers.construct_residual(inputs, indices, 1)?;

        Ok(NeuraResidual {
            layers,
            initial_offsets: self.initial_offsets,
        })
    }
}
