use super::*;

pub trait NeuraSequentialConstruct {
    type Constructed;
    type Err;

    fn construct(self, input_shape: NeuraShape) -> Result<Self::Constructed, Self::Err>;
}

#[derive(Debug, Clone)]
pub enum NeuraSequentialConstructErr<Err, ChildErr> {
    Current(Err),
    Child(ChildErr),
}

impl<Layer: NeuraPartialLayer> NeuraSequentialConstruct for NeuraSequential<Layer, ()> {
    type Constructed = NeuraSequential<Layer::Constructed, ()>;
    type Err = Layer::Err;

    fn construct(self, input_shape: NeuraShape) -> Result<Self::Constructed, Self::Err> {
        Ok(NeuraSequential {
            layer: self.layer.construct(input_shape)?,
            child_network: Box::new(()),
        })
    }
}

impl<Layer: NeuraPartialLayer, ChildNetwork: NeuraSequentialConstruct> NeuraSequentialConstruct
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
            .construct(Layer::output_shape(&layer))
            .map_err(|e| NeuraSequentialConstructErr::Child(e))?;
        let child_network = Box::new(child_network);

        Ok(NeuraSequential {
            layer,
            child_network,
        })
    }
}
