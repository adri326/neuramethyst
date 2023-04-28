pub mod algebra;
pub mod derivable;
pub mod gradient_solver;
pub mod layer;
pub mod network;
pub mod train;

mod utils;

// TODO: move to a different file
pub use utils::{argmax, cycle_shuffling, one_hot, plot_losses};

/// Common traits and structs that are useful to use this library.
/// All of these traits are prefixed with the word "neura" in some way,
/// so there should not be any conflicts when doing a wildcard import of `prelude`.
pub mod prelude {
    // Macros
    pub use crate::{neura_layer, neura_residual, neura_sequential};

    // Structs and traits
    pub use crate::gradient_solver::NeuraBackprop;
    pub use crate::layer::{
        NeuraLayer, NeuraPartialLayer, NeuraShape, NeuraTrainableLayerBackprop,
        NeuraTrainableLayerBase, NeuraTrainableLayerSelf,
    };
    pub use crate::network::sequential::{
        NeuraSequential, NeuraSequentialConstruct, NeuraSequentialLock, NeuraSequentialTail,
    };
    pub use crate::train::NeuraBatchedTrainer;
}
