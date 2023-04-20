#![feature(generic_arg_infer)]
// #![feature(generic_const_exprs)]

pub mod algebra;
pub mod derivable;
pub mod layer;
pub mod network;
pub mod optimize;
pub mod train;

mod utils;

// TODO: move to a different file
pub use utils::{argmax, cycle_shuffling, one_hot};

pub mod prelude {
    // Macros
    pub use crate::{neura_layer, neura_sequential};

    // Structs and traits
    pub use crate::layer::*;
    pub use crate::network::sequential::{
        NeuraSequential, NeuraSequentialConstruct, NeuraSequentialTail,
    };
    pub use crate::optimize::NeuraBackprop;
    pub use crate::train::NeuraBatchedTrainer;
}
