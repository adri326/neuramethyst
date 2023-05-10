use crate::err::NeuraRecursiveErr;

use super::*;

// impl<Layer: NeuraPartialLayer> NeuraPartialLayer for NeuraSequential<Layer, ()> {
//     type Constructed = NeuraSequential<Layer::Constructed, ()>;
//     type Err = Layer::Err;

//     fn construct(self, input_shape: NeuraShape) -> Result<Self::Constructed, Self::Err> {
//         Ok(NeuraSequential {
//             layer: self.layer.construct(input_shape)?,
//             child_network: Box::new(()),
//         })
//     }
// }

impl<Layer: NeuraPartialLayer, ChildNetwork: NeuraPartialLayer> NeuraPartialLayer
    for NeuraSequential<Layer, ChildNetwork>
{
    type Constructed = NeuraSequential<Layer::Constructed, ChildNetwork::Constructed>;
    type Err = NeuraRecursiveErr<Layer::Err, ChildNetwork::Err>;

    fn construct(self, input_shape: NeuraShape) -> Result<Self::Constructed, Self::Err> {
        let layer = self
            .layer
            .construct(input_shape)
            .map_err(NeuraRecursiveErr::Current)?;

        // TODO: ensure that this operation (and all recursive operations) are directly allocated on the heap
        let child_network = self
            .child_network
            .construct(layer.output_shape())
            .map_err(NeuraRecursiveErr::Child)?;
        let child_network = Box::new(child_network);

        Ok(NeuraSequential {
            layer,
            child_network,
        })
    }
}
