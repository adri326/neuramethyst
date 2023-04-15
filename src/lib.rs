#![feature(generic_arg_infer)]

pub mod algebra;
pub mod derivable;
pub mod layer;
pub mod network;
pub mod train;

mod utils;

pub mod prelude {
    // Macros
    pub use crate::{neura_layer, neura_network};

    // Structs and traits
    pub use crate::layer::{NeuraDenseLayer, NeuraDropoutLayer, NeuraLayer};
    pub use crate::network::NeuraNetwork;
    pub use crate::train::{NeuraBackprop, NeuraBatchedTrainer};
    pub use crate::utils::cycle_shuffling;
}
