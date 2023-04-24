use super::*;

pub trait NeuraResidualConstruct {
    type Constructed;
    type Err;

    fn construct_residual(
        self,
        input: NeuraResidualInput<NeuraShape>,
    ) -> Result<Self::Constructed, Self::Err>;
}

#[derive(Clone, Debug)]
pub enum NeuraResidualConstructErr<LayerErr, ChildErr> {
    LayerErr(LayerErr),
    ChildErr(ChildErr),
    OOBConnection(usize),
    AxisErr(NeuraAxisErr),
}

use NeuraResidualConstructErr::*;

impl<Layer: NeuraPartialLayer, Axis> NeuraResidualConstruct for NeuraResidualNode<Layer, (), Axis>
where
    Axis: NeuraCombineInputs<NeuraShape, Combined = Result<NeuraShape, NeuraAxisErr>>,
{
    type Constructed = NeuraResidualNode<Layer::Constructed, (), Axis>;
    type Err = NeuraResidualConstructErr<Layer::Err, ()>;

    fn construct_residual(
        self,
        input: NeuraResidualInput<NeuraShape>,
    ) -> Result<Self::Constructed, Self::Err> {
        let (layer_input_shape, _rest) = input.shift();
        let layer_input_shape = self
            .axis
            .combine(layer_input_shape)
            .map_err(|e| AxisErr(e))?;

        let layer = self
            .layer
            .construct(layer_input_shape)
            .map_err(|e| LayerErr(e))?;
        let layer_shape = layer.output_shape();

        if let Some(oob_offset) = self.offsets.iter().copied().find(|o| *o > 0) {
            return Err(OOBConnection(oob_offset));
        }
        // TODO: check rest for non-zero columns

        Ok(NeuraResidualNode {
            layer,
            child_network: (),
            offsets: self.offsets,
            axis: self.axis,
            output_shape: Some(layer_shape),
        })
    }
}

impl<Layer: NeuraPartialLayer, ChildNetwork: NeuraResidualConstruct, Axis> NeuraResidualConstruct
    for NeuraResidualNode<Layer, ChildNetwork, Axis>
where
    Axis: NeuraCombineInputs<NeuraShape, Combined = Result<NeuraShape, NeuraAxisErr>>,
{
    type Constructed = NeuraResidualNode<Layer::Constructed, ChildNetwork::Constructed, Axis>;
    type Err = NeuraResidualConstructErr<Layer::Err, ChildNetwork::Err>;

    fn construct_residual(
        self,
        input: NeuraResidualInput<NeuraShape>,
    ) -> Result<Self::Constructed, Self::Err> {
        let (layer_input_shape, mut rest) = input.shift();
        let layer_input_shape = self
            .axis
            .combine(layer_input_shape)
            .map_err(|e| AxisErr(e))?;

        let layer = self
            .layer
            .construct(layer_input_shape)
            .map_err(|e| LayerErr(e))?;
        let layer_shape = Rc::new(layer.output_shape());

        for &offset in &self.offsets {
            rest.push(offset, Rc::clone(&layer_shape));
        }
        let layer_shape = *layer_shape;

        let child_network = self
            .child_network
            .construct_residual(rest)
            .map_err(|e| ChildErr(e))?;

        Ok(NeuraResidualNode {
            layer,
            child_network,
            offsets: self.offsets,
            axis: self.axis,
            output_shape: Some(layer_shape),
        })
    }
}

impl<Layer, Axis> NeuraShapedLayer for NeuraResidualNode<Layer, (), Axis> {
    #[inline(always)]
    fn output_shape(&self) -> NeuraShape {
        self.output_shape.unwrap()
    }
}

impl<Layer, ChildNetwork: NeuraShapedLayer, Axis> NeuraShapedLayer
    for NeuraResidualNode<Layer, ChildNetwork, Axis>
{
    #[inline(always)]
    fn output_shape(&self) -> NeuraShape {
        self.child_network.output_shape()
    }
}

impl<Layers: NeuraShapedLayer> NeuraShapedLayer for NeuraResidual<Layers> {
    #[inline(always)]
    fn output_shape(&self) -> NeuraShape {
        self.layers.output_shape()
    }
}

impl<Layers: NeuraResidualConstruct> NeuraPartialLayer for NeuraResidual<Layers>
where
    // Should always be satisfied:
    Layers::Constructed: NeuraShapedLayer,
{
    type Constructed = NeuraResidual<Layers::Constructed>;
    type Err = Layers::Err;

    fn construct(self, input_shape: NeuraShape) -> Result<Self::Constructed, Self::Err> {
        let input_shape = Rc::new(input_shape);
        let mut inputs = NeuraResidualInput::new();

        for &offset in &self.initial_offsets {
            inputs.push(offset, Rc::clone(&input_shape));
        }

        drop(input_shape);

        let layers = self.layers.construct_residual(inputs)?;

        Ok(NeuraResidual {
            layers,
            initial_offsets: self.initial_offsets,
        })
    }
}
