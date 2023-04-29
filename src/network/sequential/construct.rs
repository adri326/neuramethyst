use crate::layer::NeuraShapedLayer;

use super::*;

#[derive(Debug, Clone)]
pub enum NeuraSequentialConstructErr<Err, ChildErr> {
    Current(Err),
    Child(ChildErr),
}

impl<Layer: NeuraPartialLayer> NeuraPartialLayer for NeuraSequential<Layer, ()> {
    type Constructed = NeuraSequential<Layer::Constructed, ()>;
    type Err = Layer::Err;

    fn construct(self, input_shape: NeuraShape) -> Result<Self::Constructed, Self::Err> {
        Ok(NeuraSequential {
            layer: self.layer.construct(input_shape)?,
            child_network: Box::new(()),
        })
    }
}

impl<Layer: NeuraPartialLayer, ChildNetwork: NeuraPartialLayer> NeuraPartialLayer
    for NeuraSequential<Layer, ChildNetwork>
{
    type Constructed = NeuraSequential<Layer::Constructed, ChildNetwork::Constructed>;
    type Err = NeuraSequentialConstructErr<Layer::Err, ChildNetwork::Err>;

    fn construct(self, input_shape: NeuraShape) -> Result<Self::Constructed, Self::Err> {
        let layer = self
            .layer
            .construct(input_shape)
            .map_err(|e| NeuraSequentialConstructErr::Current(e))?;

        // TODO: ensure that this operation (and all recursive operations) are directly allocated on the heap
        let child_network = self
            .child_network
            .construct(layer.output_shape())
            .map_err(|e| NeuraSequentialConstructErr::Child(e))?;
        let child_network = Box::new(child_network);

        Ok(NeuraSequential {
            layer,
            child_network,
        })
    }
}

impl<Layer: NeuraShapedLayer> NeuraShapedLayer for NeuraSequential<Layer, ()> {
    #[inline(always)]
    fn output_shape(&self) -> NeuraShape {
        self.layer.output_shape()
    }
}

impl<Layer, ChildNetwork: NeuraShapedLayer> NeuraShapedLayer
    for NeuraSequential<Layer, ChildNetwork>
{
    #[inline(always)]
    fn output_shape(&self) -> NeuraShape {
        self.child_network.output_shape()
    }
}
