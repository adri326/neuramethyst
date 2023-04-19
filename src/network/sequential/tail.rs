use super::*;

/// Operations on the tail end of a sequential network
pub trait NeuraSequentialTail {
    type TailTrimmed;
    type TailPushed<T>;

    fn trim_tail(self) -> Self::TailTrimmed;
    fn push_tail<T>(self, layer: T) -> Self::TailPushed<T>;
}

// Trimming the last layer returns an empty network
impl<Layer> NeuraSequentialTail for NeuraSequential<Layer, ()> {
    type TailTrimmed = ();
    // GAT :3
    type TailPushed<T> = NeuraSequential<Layer, NeuraSequential<T, ()>>;

    fn trim_tail(self) -> Self::TailTrimmed {
        ()
    }

    fn push_tail<T>(self, layer: T) -> Self::TailPushed<T> {
        NeuraSequential {
            layer: self.layer,
            child_network: Box::new(NeuraSequential {
                layer,
                child_network: Box::new(()),
            }),
        }
    }
}

// Trimming another layer returns a network which calls trim recursively
impl<Layer, ChildNetwork: NeuraSequentialTail> NeuraSequentialTail
    for NeuraSequential<Layer, ChildNetwork>
{
    type TailTrimmed = NeuraSequential<Layer, <ChildNetwork as NeuraSequentialTail>::TailTrimmed>;
    type TailPushed<T> =
        NeuraSequential<Layer, <ChildNetwork as NeuraSequentialTail>::TailPushed<T>>;

    fn trim_tail(self) -> Self::TailTrimmed {
        NeuraSequential {
            layer: self.layer,
            child_network: Box::new(self.child_network.trim_tail()),
        }
    }

    fn push_tail<T>(self, layer: T) -> Self::TailPushed<T> {
        NeuraSequential {
            layer: self.layer,
            child_network: Box::new(self.child_network.push_tail(layer)),
        }
    }
}
