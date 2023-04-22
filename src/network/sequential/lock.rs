use crate::layer::lock::NeuraLockLayer;

use super::*;

pub trait NeuraSequentialLock {
    type Locked;

    fn lock(self) -> Self::Locked;
}

impl NeuraSequentialLock for () {
    type Locked = ();

    fn lock(self) -> Self::Locked {
        ()
    }
}

impl<Layer, ChildNetwork: NeuraSequentialLock> NeuraSequentialLock
    for NeuraSequential<Layer, ChildNetwork>
{
    type Locked = NeuraSequential<NeuraLockLayer<Layer>, ChildNetwork::Locked>;

    fn lock(self) -> Self::Locked {
        let Self {
            layer,
            child_network,
        } = self;

        NeuraSequential {
            layer: NeuraLockLayer::new(layer),
            child_network: Box::new(child_network.lock()),
        }
    }
}
