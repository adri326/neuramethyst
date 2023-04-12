#![feature(generic_arg_infer)]

pub mod algebra;
pub mod derivable;
pub mod layer;
pub mod network;
pub mod train;

mod utils;

pub mod prelude {
    // Macros
    pub use crate::{neura_network, neura_layer};

    // Structs and traits
    pub use crate::network::{NeuraNetwork};
    pub use crate::layer::{NeuraLayer, NeuraDenseLayer};
    pub use crate::train::{NeuraBackprop, NeuraBatchedTrainer};
    pub use crate::utils::cycle_shuffling;
}
