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
    pub use super::network::{NeuraNetwork};
    pub use super::layer::{NeuraLayer, NeuraDenseLayer};
    pub use super::train::{NeuraBackprop, train_batched};
}
